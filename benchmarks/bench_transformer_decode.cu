#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <nvbench/nvbench.cuh>

#include "minillm/model/model_config_loader.h"
#include "minillm/model/safetensors_reader.h"
#include "minillm/model/transformer_model.h"
#include "minillm/model/weight_loader.h"

namespace
{

    struct ModelBenchmarkContext
    {
        minillm::ModelConfig config;
        minillm::TransformerModel model;

        explicit ModelBenchmarkContext(
            const std::filesystem::path &model_dir)
            : config(
                  minillm::load_model_config(
                      model_dir / "config.json",
                      std::min<std::size_t>(
                          minillm::load_model_config(
                              model_dir / "config.json",
                              1)
                              .max_position_embeddings,
                          2048))),
              model(config)
        {
            minillm::SafeTensorsReader reader(
                model_dir / "model.safetensors");
            minillm::load_model_weights(
                model,
                config,
                reader);
        }
    };

    const std::filesystem::path &model_directory()
    {
        static const std::filesystem::path path = []
        {
            if (const char *value =
                    std::getenv("MINILLM_MODEL_DIR"))
            {
                if (*value != '\0')
                    return std::filesystem::path(value);
            }

            return std::filesystem::path(
                "models/SmolLM2-135M-Instruct");
        }();

        return path;
    }

    ModelBenchmarkContext &context()
    {
        static std::unique_ptr<ModelBenchmarkContext> value;
        if (!value)
        {
            const auto directory = model_directory();
            if (!std::filesystem::is_regular_file(
                    directory / "config.json") ||
                !std::filesystem::is_regular_file(
                    directory / "model.safetensors"))
            {
                throw std::runtime_error(
                    "Model files were not found in " +
                    directory.string() +
                    ". Set MINILLM_MODEL_DIR to the model directory.");
            }

            value = std::make_unique<ModelBenchmarkContext>(directory);
        }
        return *value;
    }

    void bench_model_decode(nvbench::state &state)
    {
        auto &model_context = context();

        const std::size_t context_length =
            static_cast<std::size_t>(
                state.get_int64("Context"));

        if (context_length + 1 >
            model_context.config.kv_capacity)
        {
            throw std::invalid_argument(
                "Benchmark context exceeds model KV capacity");
        }

        minillm::DecodeSession session(model_context.config);
        minillm::ModelWorkspace workspace(model_context.config);

        // Prefill a deterministic, non-special-token prefix. It is outside
        // the timed region because this benchmark measures one decode step.
        for (std::size_t position = 0;
             position < context_length;
             ++position)
        {
            const std::size_t token_id =
                3 + position % (model_context.config.vocab_size - 3);

            model_context.model.forward_decode(
                token_id,
                position,
                session,
                workspace);
        }

        const std::size_t next_token = 3;

        state.exec(
            nvbench::exec_tag::sync,
            [&](nvbench::launch &launch)
            {
                // NVBench repeats the timed region. Reuse the
                // prefetched KV prefix for every sample.
                session.rewind_to(context_length);

                model_context.model.forward_decode_async(
                    next_token,
                    context_length,
                    session,
                    workspace,
                    launch.get_stream());
            });
    }

} // namespace

NVBENCH_BENCH(bench_model_decode)
    .set_name("model_decode")
    .add_int64_axis("Context", {1, 32, 128, 256});
