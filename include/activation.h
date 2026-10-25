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

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

typedef enum ActivationType {
    ACT_NONE = 0,
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_TANH,
    ACT_SIGMOID,
    ACT_SOFTMAX_CEL
} ActivationType;

void activation_forward(Tensor *t, ActivationType activation);

#endif
