
#include "config/model_registry.h"

#include <fstream>
#include <stdexcept>

namespace minillm
{

    ModelRegistry::ModelRegistry(
        const std::filesystem::path &registry_file)
        : registry_file_(
              std::filesystem::absolute(registry_file)
                  .lexically_normal())
    {
        std::ifstream file(registry_file_);

        if (!file)
        {
            throw std::runtime_error(
                "Cannot open model registry: " +
                registry_file_.string());
        }

        file >> registry_;

        if (registry_.at("schema_version").get<int>() != 1)
        {
            throw std::runtime_error(
                "Unsupported model registry version");
        }

        if (!registry_.at("models").is_object())
        {
            throw std::runtime_error(
                "models must be a JSON object");
        }
    }

    ModelEntry ModelRegistry::resolve(
        const std::string &name) const
    {
        const std::string selected =
            name.empty()
                ? registry_.at("default_model")
                      .get<std::string>()
                : name;

        const auto &models = registry_.at("models");

        if (!models.contains(selected))
        {
            throw std::runtime_error(
                "Unknown model: " + selected);
        }

        const auto &entry = models.at(selected);

        if (entry.value("format", std::string{}) !=
            "safetensors")
        {
            throw std::runtime_error(
                "Unsupported weight format");
        }

        const auto relative_dir =
            std::filesystem::path(
                entry.at("path").get<std::string>());

        // 相对于 models.json 进行路径解析
        const auto model_dir =
            (registry_file_.parent_path() / relative_dir)
                .lexically_normal();

        const auto weights_file =
            (model_dir /
             entry.at("weight_file").get<std::string>())
                .lexically_normal();

        const auto config_file =
            model_dir / "config.json";

        if (!std::filesystem::is_regular_file(config_file))
        {
            throw std::runtime_error(
                "Missing config.json: " +
                config_file.string());
        }

        if (!std::filesystem::is_regular_file(weights_file))
        {
            throw std::runtime_error(
                "Missing model weights: " +
                weights_file.string());
        }

        return ModelEntry{
            selected,
            model_dir,
            config_file,
            weights_file};
    }

} // namespace minillm
