#pragma once

#include <cstddef>
#include <filesystem>

#include "minillm/model/model_config.h"

namespace minillm
{

    ModelConfig load_model_config(
        const std::filesystem::path &config_path,
        std::size_t kv_capacity);

}