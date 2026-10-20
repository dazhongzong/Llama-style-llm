#include "minillm/generation/tokenizer.h"

#include <tokenizers_cpp.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace minillm
{

    namespace
    {
        using Json = nlohmann::json;

        std::string read_file(const std::filesystem::path &path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                throw std::runtime_error(
                    "Cannot open tokenizer file: " + path.string());
            }

            return std::string(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>());
        }

        Json read_json_if_present(const std::filesystem::path &path)
        {
            if (!std::filesystem::is_regular_file(path))
                return Json::object();

            std::ifstream file(path);
            if (!file)
                throw std::runtime_error(
                    "Cannot open tokenizer config: " + path.string());

            Json result;
            file >> result;
            return result;
        }

        std::size_t special_token_id(
            tokenizers::Tokenizer &tokenizer,
            const Json &config,
            const char *config_key,
            std::size_t fallback)
        {
            if (!config.contains(config_key) ||
                !config.at(config_key).is_string())
            {
                return fallback;
            }

            const auto id = tokenizer.TokenToId(
                config.at(config_key).get<std::string>());
            if (id < 0)
            {
                throw std::runtime_error(
                    std::string("Tokenizer config token is missing from vocabulary: ") +
                    config.at(config_key).get<std::string>());
            }

            return static_cast<std::size_t>(id);
        }
    } // namespace

    Tokenizer::Tokenizer(
        const std::filesystem::path &tokenizer_path)
        : tokenizer_(nullptr)
    {
        const std::string tokenizer_blob = read_file(tokenizer_path);
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);
        if (!tokenizer_)
            throw std::runtime_error("tokenizers-cpp returned a null tokenizer");

        const Json tokenizer_json = Json::parse(tokenizer_blob);
        if (tokenizer_json.contains("added_tokens"))
        {
            for (const auto &token : tokenizer_json.at("added_tokens"))
            {
                if (token.value("special", false))
                {
                    special_token_ids_.insert(
                        token.at("id").get<std::size_t>());
                }
            }
        }

        const Json config = read_json_if_present(
            tokenizer_path.parent_path() / "tokenizer_config.json");

        eos_token_id_ = special_token_id(
            *tokenizer_, config, "eos_token", eos_token_id_);
        bos_token_id_ = special_token_id(
            *tokenizer_, config, "bos_token", bos_token_id_);
    }

    Tokenizer::~Tokenizer() = default;

    std::vector<std::size_t> Tokenizer::encode(
        const std::string &text) const
    {
        // tokenizers-cpp's Encode API is not const-qualified.
        auto &tokenizer = *tokenizer_;
        const std::vector<std::int32_t> ids = tokenizer.Encode(text);

        std::vector<std::size_t> result;
        result.reserve(ids.size());
        for (const std::int32_t id : ids)
        {
            if (id < 0)
                throw std::runtime_error("Tokenizer produced a negative token id");
            result.push_back(static_cast<std::size_t>(id));
        }
        return result;
    }

    std::string Tokenizer::decode(
        const std::vector<std::size_t> &ids,
        bool skip_special_tokens) const
    {
        std::vector<std::int32_t> token_ids;
        token_ids.reserve(ids.size());
        for (const std::size_t id : ids)
        {
            if (id > static_cast<std::size_t>(
                        std::numeric_limits<std::int32_t>::max()))
            {
                throw std::out_of_range("Token id does not fit int32_t");
            }
            if (skip_special_tokens && special_token_ids_.contains(id))
                continue;

            token_ids.push_back(static_cast<std::int32_t>(id));
        }

        auto &tokenizer = *tokenizer_;
        return tokenizer.Decode(token_ids);
    }

    std::size_t Tokenizer::vocab_size() const noexcept
    {
        return tokenizer_->GetVocabSize();
    }

    bool Tokenizer::has_special_token(const std::string &text) const
    {
        const auto id = tokenizer_->TokenToId(text);
        return id >= 0 && special_token_ids_.contains(
                               static_cast<std::size_t>(id));
    }

} // namespace minillm
