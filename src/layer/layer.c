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

#include "layer/layer.h"

Error layer_init(Layer *layer, const LayerConfig *config) {
    if (!layer || !config)
        return ERROR_INVALID_ARG;

    layer->layer_type      = config->layer_type;
    layer->activation_type = config->activation_type;

    Error err = ERROR_OK;
    switch (config->layer_type) {
        case LAYER_DENSE: {
            err = dense_init(&layer->dense_layer, &config->dense_config);
            if (err != ERROR_OK)
                return err;

            break;
        }

        default:
            return ERROR_UNKOWN_TYPE;
    };

    return ERROR_OK;
}

void layer_destroy(Layer *layer) {
    switch (layer->layer_type) {
        case LAYER_DENSE:
            dense_destroy(&layer->dense_layer);
            break;

        default:
    }    
}

Error layer_forward(Tensor *output, Layer *layer, const Tensor *input) {
    Error err = ERROR_OK;
    switch (layer->layer_type) {
        case LAYER_DENSE: {
            err = dense_forward(output, &layer->dense_layer, input);
            if (err != ERROR_OK)
                return err;

            break;
        }

        default:
            return ERROR_UNKOWN_TYPE;
    };

    return ERROR_OK;
}