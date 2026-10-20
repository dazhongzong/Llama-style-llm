#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace tokenizers
{
    class Tokenizer;
}

namespace minillm
{

    // Thin project-level adapter around tokenizers-cpp.  The model runtime
    // uses size_t token ids, while tokenizers-cpp exposes int32_t ids.
    class Tokenizer
    {
    public:
        explicit Tokenizer(const std::filesystem::path &tokenizer_path);
        ~Tokenizer();

        std::vector<std::size_t> encode(const std::string &text) const;

        std::string decode(
            const std::vector<std::size_t> &ids,
            bool skip_special_tokens = true) const;

        std::size_t eos_token_id() const noexcept
        {
            return eos_token_id_;
        }

        std::size_t bos_token_id() const noexcept
        {
            return bos_token_id_;
        }

        std::size_t vocab_size() const noexcept;

        bool has_special_token(const std::string &text) const;

    private:
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
        std::size_t eos_token_id_ = 2;
        std::size_t bos_token_id_ = 1;
        std::unordered_set<std::size_t> special_token_ids_;
    };

} // namespace minillm
