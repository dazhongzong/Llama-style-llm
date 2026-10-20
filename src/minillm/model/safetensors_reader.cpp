
#include "minillm/model/safetensors_reader.h"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace minillm
{

    namespace
    {

        std::uint16_t read_u16_le(const unsigned char *p)
        {
            return std::uint16_t(p[0]) |
                   (std::uint16_t(p[1]) << 8);
        }

        std::uint32_t read_u32_le(const unsigned char *p)
        {
            return std::uint32_t(p[0]) |
                   (std::uint32_t(p[1]) << 8) |
                   (std::uint32_t(p[2]) << 16) |
                   (std::uint32_t(p[3]) << 24);
        }

        std::uint64_t read_u64_le(const unsigned char *p)
        {
            std::uint64_t result = 0;

            for (int i = 0; i < 8; ++i)
            {
                result |= std::uint64_t(p[i]) << (i * 8);
            }

            return result;
        }

        float bf16_to_float(std::uint16_t bits)
        {
            const std::uint32_t fp32_bits =
                std::uint32_t(bits) << 16;

            return std::bit_cast<float>(fp32_bits);
        }

    } // namespace

    SafeTensorsReader::SafeTensorsReader(
        const std::filesystem::path &path)
        : path_(path)
    {
        std::ifstream file(path_, std::ios::binary);

        if (!file)
            throw std::runtime_error(
                "Cannot open SafeTensors file");

        const std::uint64_t file_size =
            std::filesystem::file_size(path_);

        if (file_size < 8)
            throw std::runtime_error(
                "SafeTensors file is too small");

        std::array<unsigned char, 8> prefix{};

        if (!file.read(
                reinterpret_cast<char *>(prefix.data()),
                8))
        {
            throw std::runtime_error(
                "Cannot read SafeTensors header length");
        }

        const std::uint64_t header_size =
            read_u64_le(prefix.data());

        // 教学版安全限制：Header 不超过 16 MiB
        constexpr std::uint64_t max_header =
            16ULL * 1024 * 1024;

        if (header_size > max_header ||
            header_size > file_size - 8 ||
            header_size == 0)
        {
            throw std::runtime_error(
                "Invalid SafeTensors header size");
        }

        std::string header_text(
            static_cast<std::size_t>(header_size), '\0');

        if (!file.read(
                header_text.data(),
                static_cast<std::streamsize>(header_size)))
        {
            throw std::runtime_error(
                "Cannot read SafeTensors JSON header");
        }

        header_ = nlohmann::json::parse(header_text);

        if (!header_.is_object())
            throw std::runtime_error(
                "Invalid SafeTensors JSON object");

        data_begin_ = 8 + header_size;
        data_size_ = file_size - data_begin_;
    }

    bool SafeTensorsReader::contains(
        const std::string &name) const
    {
        return header_.contains(name);
    }

    HostWeightF32 SafeTensorsReader::read_f32(
        const std::string &name) const
    {
        if (!contains(name))
            throw std::runtime_error(
                "Missing weight: " + name);

        const auto &meta = header_.at(name);

        const std::string dtype =
            meta.at("dtype").get<std::string>();

        if (dtype != "BF16" && dtype != "F32")
            throw std::runtime_error(
                "Unsupported weight dtype: " + dtype);

        HostWeightF32 result;

        result.shape =
            meta.at("shape").get<std::vector<std::size_t>>();

        // 当前 MiniLLM 的 Tensor 不使用空 Shape
        if (result.shape.empty())
            throw std::runtime_error(
                "Unsupported scalar weight");

        std::size_t elements = 1;

        for (const std::size_t dim : result.shape)
        {
            if (dim == 0 ||
                elements >
                    std::numeric_limits<std::size_t>::max() / dim)
            {
                throw std::runtime_error(
                    "Invalid weight shape");
            }

            elements *= dim;
        }

        const auto offsets =
            meta.at("data_offsets")
                .get<std::vector<std::uint64_t>>();

        if (offsets.size() != 2)
            throw std::runtime_error(
                "Invalid weight offsets");

        const std::uint64_t begin = offsets[0];
        const std::uint64_t end = offsets[1];

        if (begin > end || end > data_size_)
            throw std::runtime_error(
                "Weight offsets outside file");

        const std::size_t bytes_per_element =
            dtype == "BF16" ? 2 : 4;

        if (elements >
            std::numeric_limits<std::size_t>::max() / bytes_per_element)
        {
            throw std::runtime_error(
                "Weight byte count overflow");
        }

        const std::size_t expected_bytes =
            elements * bytes_per_element;

        if (end - begin != expected_bytes)
            throw std::runtime_error(
                "Weight size does not match Shape");

        if (expected_bytes >
                static_cast<std::size_t>(
                    std::numeric_limits<std::streamsize>::max()) ||
            data_begin_ + begin >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max()))
        {
            throw std::runtime_error(
                "Weight too large for file stream");
        }

        std::ifstream file(path_, std::ios::binary);

        if (!file)
            throw std::runtime_error(
                "Cannot reopen SafeTensors file");

        const auto absolute_offset =
            static_cast<std::streamoff>(data_begin_ + begin);

        file.seekg(absolute_offset, std::ios::beg);

        if (!file)
            throw std::runtime_error(
                "Cannot seek to weight");

        std::vector<unsigned char> raw(expected_bytes);

        if (!file.read(
                reinterpret_cast<char *>(raw.data()),
                static_cast<std::streamsize>(raw.size())))
        {
            throw std::runtime_error(
                "Cannot read weight data");
        }

        result.values.resize(elements);

        for (std::size_t i = 0; i < elements; ++i)
        {
            const unsigned char *p =
                raw.data() + i * bytes_per_element;

            if (dtype == "BF16")
            {
                result.values[i] =
                    bf16_to_float(read_u16_le(p));
            }
            else
            {
                const std::uint32_t bits =
                    read_u32_le(p);

                result.values[i] =
                    std::bit_cast<float>(bits);
            }
        }

        return result;
    }

} // namespace minillm
