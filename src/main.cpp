#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/model_registry.h"
#include "minillm/generation/tokenizer.h"
#include "minillm/model/model_config_loader.h"
#include "minillm/model/safetensors_reader.h"
#include "minillm/model/transformer_model.h"
#include "minillm/model/weight_loader.h"

namespace
{

    struct GenerationOptions
    {
        std::string model_name;
        std::string prompt =
            "Explain how CUDA kernels are launched.";
        std::size_t max_new_tokens = 64;
        bool raw_prompt = false;
        bool chat = false;
    };

    struct ChatMessage
    {
        std::string role;
        std::string content;
    };

    struct GenerationResult
    {
        std::string text;
        std::size_t token_count = 0;
    };

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program << " [options]\n"
            << "\nOptions:\n"
            << "  --model NAME             Model name in configs/models.json\n"
            << "  --prompt TEXT            User prompt; use '-' to read stdin\n"
            << "  --max-new-tokens N       Maximum generated tokens (default: 64)\n"
            << "  --raw                    Do not apply the chat template\n"
            << "  --chat                   Start an interactive conversation\n"
            << "  --help                   Show this help\n";
    }

    std::size_t parse_size(
        const std::string &value,
        const char *option)
    {
        try
        {
            std::size_t consumed = 0;
            const auto result = std::stoull(value, &consumed);
            if (consumed != value.size())
                throw std::invalid_argument("trailing characters");
            return static_cast<std::size_t>(result);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(
                std::string("Invalid value for ") + option +
                ": " + value);
        }
    }

    GenerationOptions parse_options(
        int argc,
        char **argv)
    {
        GenerationOptions options;

        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];

            if (argument == "--help" || argument == "-h")
            {
                print_usage(argv[0]);
                std::exit(0);
            }
            if (argument == "--raw")
            {
                options.raw_prompt = true;
                continue;
            }
            if (argument == "--chat")
            {
                options.chat = true;
                continue;
            }

            auto require_value = [&](const char *option)
            {
                if (i + 1 >= argc)
                    throw std::invalid_argument(
                        std::string("Missing value for ") + option);
                return std::string(argv[++i]);
            };

            if (argument == "--model")
            {
                options.model_name = require_value("--model");
            }
            else if (argument == "--prompt")
            {
                options.prompt = require_value("--prompt");
            }
            else if (argument == "--max-new-tokens")
            {
                options.max_new_tokens = parse_size(
                    require_value("--max-new-tokens"),
                    "--max-new-tokens");
            }
            else
            {
                throw std::invalid_argument(
                    "Unknown option: " + argument);
            }
        }

        return options;
    }

    std::filesystem::path find_registry_path(
        const char *program)
    {
        const std::filesystem::path from_working_directory =
            "configs/models.json";
        if (std::filesystem::is_regular_file(
                from_working_directory))
            return from_working_directory;

        // Also allow running build/app from outside the project root.
        const auto executable =
            std::filesystem::absolute(program);
        const auto from_build_directory =
            executable.parent_path().parent_path() /
            "configs/models.json";

        if (std::filesystem::is_regular_file(
                from_build_directory))
            return from_build_directory;

        throw std::runtime_error(
            "Cannot find configs/models.json");
    }

    std::string make_prompt(
        const minillm::Tokenizer &tokenizer,
        const std::string &user_prompt,
        bool raw_prompt)
    {
        if (raw_prompt)
            return user_prompt;

        if (tokenizer.has_special_token("<|im_start|>"))
        {
            return "<|im_start|>system\n"
                   "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
                   "<|im_end|>\n"
                   "<|im_start|>user\n" +
                   user_prompt +
                   "<|im_end|>\n"
                   "<|im_start|>assistant\n";
        }

        // TinyLlama's tokenizer_config.json uses the Llama chat template.
        return "<|user|>\n" +
               user_prompt +
                   "</s>\n<|assistant|>\n";
    }

    std::string make_chat_prompt(
        const minillm::Tokenizer &tokenizer,
        const std::vector<ChatMessage> &history)
    {
        std::string prompt;

        if (tokenizer.has_special_token("<|im_start|>"))
        {
            prompt =
                "<|im_start|>system\n"
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
                "<|im_end|>\n";

            for (const auto &message : history)
            {
                prompt += "<|im_start|>" + message.role + "\n";
                prompt += message.content;
                prompt += "<|im_end|>\n";
            }
            prompt += "<|im_start|>assistant\n";
            return prompt;
        }

        for (const auto &message : history)
        {
            if (message.role == "user")
                prompt += "<|user|>\n";
            else
                prompt += "<|assistant|>\n";
            prompt += message.content;
            prompt += "</s>\n";
        }
        prompt += "<|assistant|>\n";
        return prompt;
    }

    std::size_t greedy_argmax(
        const minillm::Tensor &cpu_logits)
    {
        const float *values = cpu_logits.ptr<float>();
        const std::size_t vocab_size =
            cpu_logits.shape().at(0);

        std::size_t best = 0;
        float best_value = -std::numeric_limits<float>::infinity();

        for (std::size_t id = 0; id < vocab_size; ++id)
        {
            if (values[id] > best_value)
            {
                best_value = values[id];
                best = id;
            }
        }
        return best;
    }

    GenerationResult generate_response(
        const minillm::TransformerModel &model,
        const minillm::ModelConfig &config,
        const minillm::Tokenizer &tokenizer,
        const std::vector<std::size_t> &prompt_tokens,
        std::size_t max_new_tokens,
        minillm::ModelWorkspace &workspace,
        minillm::Tensor &cpu_logits)
    {
        minillm::DecodeSession session(config);

        for (std::size_t position = 0;
             position < prompt_tokens.size();
             ++position)
        {
            model.forward_decode(
                prompt_tokens[position],
                position,
                session,
                workspace);
        }

        std::vector<std::size_t> generated;
        generated.reserve(max_new_tokens);

        for (std::size_t step = 0;
             step < max_new_tokens;
             ++step)
        {
            cpu_logits.copy_from(workspace.logits);
            const std::size_t next_token =
                greedy_argmax(cpu_logits);

            if (next_token == tokenizer.eos_token_id())
                break;

            generated.push_back(next_token);

            model.forward_decode(
                next_token,
                prompt_tokens.size() + step,
                session,
                workspace);
        }

        return GenerationResult{
            tokenizer.decode(generated),
            generated.size()};
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        using namespace minillm;

        const GenerationOptions options =
            parse_options(argc, argv);

        GenerationOptions resolved_options = options;
        if (!resolved_options.chat &&
            resolved_options.prompt == "-")
        {
            if (!std::getline(std::cin, resolved_options.prompt) ||
                resolved_options.prompt.empty())
            {
                throw std::invalid_argument(
                    "stdin prompt is empty");
            }
        }

        const auto registry_path =
            find_registry_path(argv[0]);
        ModelRegistry registry(registry_path);
        const ModelEntry selected =
            registry.resolve(resolved_options.model_name);

        std::cout << "[MODEL] " << selected.name << '\n'
                  << "[PATH] " << selected.model_dir << '\n';

        Tokenizer tokenizer(
            selected.model_dir / "tokenizer.json");

        const ModelConfig metadata =
            load_model_config(selected.config_file, 1);

        std::size_t kv_capacity =
            metadata.max_position_embeddings;
        if (!resolved_options.chat)
        {
            const std::string formatted_prompt =
                make_prompt(
                    tokenizer,
                    resolved_options.prompt,
                    resolved_options.raw_prompt);
            const std::vector<std::size_t> prompt_tokens =
                tokenizer.encode(formatted_prompt);

            if (prompt_tokens.empty())
                throw std::runtime_error("Prompt produced no tokens");

            const std::size_t required_tokens =
                prompt_tokens.size() + resolved_options.max_new_tokens;
            if (required_tokens > metadata.max_position_embeddings)
            {
                throw std::invalid_argument(
                    "Prompt and generation length exceed model context: " +
                    std::to_string(required_tokens) + " > " +
                    std::to_string(metadata.max_position_embeddings));
            }

            kv_capacity = std::min(
                metadata.max_position_embeddings,
                std::max<std::size_t>(256, required_tokens));
        }

        if (resolved_options.chat && resolved_options.raw_prompt)
            throw std::invalid_argument(
                "--chat cannot be combined with --raw");

        const ModelConfig config =
            load_model_config(selected.config_file, kv_capacity);

        TransformerModel model(config);
        SafeTensorsReader reader(selected.weights_file);
        load_model_weights(model, config, reader);

        DecodeSession session(config);
        ModelWorkspace workspace(config);
        Tensor cpu_logits(
            {config.vocab_size},
            DataType::FLOAT32,
            DeviceType::CPU);

        if (resolved_options.chat)
        {
            std::vector<ChatMessage> history;
            std::cout << "[CHAT] Type /exit or /quit to stop, "
                         "/reset to clear history.\n";

            while (true)
            {
                std::cout << "You> " << std::flush;
                std::string user_text;
                if (!std::getline(std::cin, user_text))
                {
                    std::cout << '\n';
                    break;
                }

                if (user_text == "/exit" ||
                    user_text == "/quit")
                    break;

                if (user_text == "/reset")
                {
                    history.clear();
                    std::cout << "[CHAT] History cleared.\n";
                    continue;
                }

                if (user_text == "/help")
                {
                    std::cout << "[CHAT] Commands: /exit, /quit, "
                                 "/reset, /help\n";
                    continue;
                }

                if (user_text.empty())
                    continue;

                history.push_back({"user", user_text});
                const std::string formatted_prompt =
                    make_chat_prompt(tokenizer, history);
                const std::vector<std::size_t> prompt_tokens =
                    tokenizer.encode(formatted_prompt);
                const std::size_t required_tokens =
                    prompt_tokens.size() + resolved_options.max_new_tokens;

                if (prompt_tokens.empty())
                {
                    history.pop_back();
                    std::cout << "[ERROR] Prompt produced no tokens.\n";
                    continue;
                }

                if (required_tokens > config.max_position_embeddings)
                {
                    history.pop_back();
                    std::cout << "[ERROR] Conversation exceeds model "
                                 "context. Use /reset to clear history.\n";
                    continue;
                }

                const GenerationResult result =
                    generate_response(
                        model,
                        config,
                        tokenizer,
                        prompt_tokens,
                        resolved_options.max_new_tokens,
                        workspace,
                        cpu_logits);
                history.push_back({"assistant", result.text});

                std::cout << "Assistant> " << result.text << '\n'
                          << "[GENERATED TOKENS] "
                          << result.token_count << '\n';
            }
        }
        else
        {
            const std::string formatted_prompt =
                make_prompt(
                    tokenizer,
                    resolved_options.prompt,
                    resolved_options.raw_prompt);
            const std::vector<std::size_t> prompt_tokens =
                tokenizer.encode(formatted_prompt);

            if (prompt_tokens.empty())
                throw std::runtime_error("Prompt produced no tokens");

            const GenerationResult result =
                generate_response(
                    model,
                    config,
                    tokenizer,
                    prompt_tokens,
                    resolved_options.max_new_tokens,
                    workspace,
                    cpu_logits);

            std::cout << "[PROMPT] " << resolved_options.prompt << '\n'
                      << "[OUTPUT] " << result.text << '\n'
                      << "[GENERATED TOKENS] "
                      << result.token_count << '\n';
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
