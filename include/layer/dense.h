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

#ifndef DENSE_H
#define DENSE_H

#include "common.h"
#include "tensor.h"

typedef struct {
    Tensor weights;
    Tensor bias;
} Dense;

typedef struct {
    usize input_size;
    usize output_size;
} DenseConfig;

Error dense_init(Dense *dense_layer, const DenseConfig *config);
void dense_destroy(Dense *dense_layer);

Error dense_forward(Tensor *output, Dense *dense_layer, const Tensor *input);

#endif /* DENSE_H */