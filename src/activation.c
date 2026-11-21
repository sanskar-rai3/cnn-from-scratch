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

#include "activation.h"

#include <math.h>

static const float LEAKY_RELU_ALPHA = 0.01f;

static void relu_(Tensor *t) {
    for (usize i = 0; i < t->size; i++) {
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
    }
}

static void leaky_relu_(Tensor *t) {
    for (usize i = 0; i < t->size; i++) {
        if (t->data[i] < 0.0f) t->data[i] *= LEAKY_RELU_ALPHA;
    }
}

static void tanh_(Tensor *t) {
    for (usize i = 0; i < t->size; i++) {
        t->data[i] = tanf(t->data[i]);
    }
}

static void sigmoid_(Tensor *t) {
    for (usize i = 0; i < t->size; i++) {
        t->data[i] = 1 / (1 + expf(-(t->data[i])));
    }
}

static void softmax_(Tensor *t) {
    float sum = 0;
    for (usize i = 0; i < t->size; i++) {
        t->data[i] = expf(t->data[i]);

        sum += t->data[i];
    }

    float inv_sum = 1.0f / (sum + 1e-7f); // Avoid div by zero

    for (usize i = 0; i < t->size; i++) {
        t->data[i] *= inv_sum;
    }
}

Error activation_forward(Tensor *t, ActivationType type) {
    if (!t)
        return ERROR_INVALID_ARG;

    switch (type) {
        case ACT_RELU:
            relu_(t);
            break;

        case ACT_LEAKY_RELU:
            leaky_relu_(t);
            break;

        case ACT_TANH:
            tanh_(t);
            break;

        case ACT_SIGMOID:
            sigmoid_(t);
            break;

        case ACT_SOFTMAX:
            softmax_(t);
            break;

        case ACT_NONE:
            break;

        default:
            return ERROR_UNKOWN_TYPE;
    }

    return ERROR_OK;
}