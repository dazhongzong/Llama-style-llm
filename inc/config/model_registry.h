#pragma once

#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

namespace minillm
{

    struct ModelEntry
    {
        std::string name;
        std::filesystem::path model_dir;
        std::filesystem::path config_file;
        std::filesystem::path weights_file;
    };

    class ModelRegistry
    {
    public:
        explicit ModelRegistry(
            const std::filesystem::path &registry_file);

        // name 为空时使用 default_model
        ModelEntry resolve(
            const std::string &name = "") const;

    private:
        std::filesystem::path registry_file_;
        nlohmann::json registry_;
    };

} // namespace minillm