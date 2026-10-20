#include <iostream>
#include <string>

#include "minillm/model/safetensors_reader.h"

int main()
{
    using namespace minillm;

    SafeTensorsReader reader(
        "../models/SmolLM2-135M-Instruct/model.safetensors");

    const std::string name =
        "model.layers.0.self_attn.k_proj.weight";

    if (!reader.contains(name))
    {
        std::cerr << "Weight not found\n";
        return 1;
    }

    HostWeightF32 weight = reader.read_f32(name);

    std::cout << "Weight: " << name << '\n';

    std::cout << "Shape: ";
    for (auto dim : weight.shape)
        std::cout << dim << " ";

    std::cout << "\nElements: "
              << weight.values.size() << '\n';

    std::cout << "First 10 values:\n";

    for (std::size_t i = 0;
         i < 10 && i < weight.values.size();
         ++i)
    {
        std::cout << weight.values[i] << "\n";
    }

    return 0;
}