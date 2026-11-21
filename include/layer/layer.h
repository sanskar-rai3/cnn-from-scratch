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

#ifndef LAYER_H
#define LAYER_H

#include "layer/dense.h"
#include "activation.h"

typedef enum {
    LAYER_DENSE = 0,
    LAYER_CONVOLUTION,
    LAYER_MAXPOOL
} LayerType;

typedef struct {
    LayerType layer_type;

    union {
        Dense dense_layer;
    };

    ActivationType activation_type;
} Layer; 

typedef struct {
    LayerType layer_type;

    union {
        DenseConfig dense_config;
    };

    ActivationType activation_type;
} LayerConfig;

Error layer_init(Layer *layer, const LayerConfig *config);
void layer_destroy(Layer *layer);

Error layer_forward(Tensor *output, Layer *layer, const Tensor *input);

#endif /* LAYER_H */