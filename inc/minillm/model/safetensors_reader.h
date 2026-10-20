#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace minillm
{

    struct HostWeightF32
    {
        std::vector<std::size_t> shape;
        std::vector<float> values;
    };

    class SafeTensorsReader
    {
    public:
        explicit SafeTensorsReader(
            const std::filesystem::path &path);

        bool contains(const std::string &name) const;

        HostWeightF32 read_f32(
            const std::string &name) const;

    private:
        std::filesystem::path path_;
        std::uint64_t data_begin_ = 0;
        std::uint64_t data_size_ = 0;
        nlohmann::json header_;
    };

} // namespace minillm