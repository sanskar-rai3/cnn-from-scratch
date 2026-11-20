/*
 * Convolutional Neural Network From Scratch in C
 *
 * Copyright (C) 2026 Sanskar Rai 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "dense.h"

Error dense_init(Dense *dense_layer, const DenseConfig *config) {
    if (!dense_layer || !config)
        return ERROR_INVALID_ARG;

    usize weight_shape[] = {
        config->input_size,
        config->output_size
    };

    usize bias_shape[] = {
        config->output_size
    };

    Error err = tensor_create(&dense_layer->weights, 2, weight_shape);
    if (err != ERROR_OK)
        goto cleanup;

    err = tensor_create(&dense_layer->bias, 1, bias_shape);
    if (err != ERROR_OK)
        goto cleanup;
    
    // Using He initialization for weights.
    err = tensor_he_uniform(&dense_layer->weights, config->input_size);
    if (err != ERROR_OK)
        goto cleanup;
    
    // Initializing bias with 0.
    err = tensor_fill(&dense_layer->bias, 0.0f);
    if (err != ERROR_OK)
        goto cleanup;

    return ERROR_OK;

cleanup:
    tensor_destroy(&dense_layer->weights);
    tensor_destroy(&dense_layer->bias);

    return err;
}

void dense_destroy(Dense *dense_layer) {
    tensor_destroy(&dense_layer->weights);
    tensor_destroy(&dense_layer->bias);
}