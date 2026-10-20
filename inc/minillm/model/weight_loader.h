#pragma once

#include "minillm/model/model_config.h"
#include "minillm/model/safetensors_reader.h"
#include "minillm/model/transformer_model.h"

namespace minillm
{

    void load_model_weights(
        TransformerModel &model,
        const ModelConfig &config,
        const SafeTensorsReader &reader);

}