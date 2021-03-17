import logging
from textwrap import dedent
from typing import Optional

import numpy as np
import tensorflow as tf

from inputtensorfi.attacks.differential_evolution import differential_evolution
from inputtensorfi.manipulation.img.utils import (
    original_perturb_image,
    original_perturb_image_by_bit_fault,
)


def predict_classes(
    xs: np.ndarray, img: np.ndarray, y_true: int, model: tf.keras.Model
) -> np.ndarray:
    """Perturb the image and get the predictions of the model."""
    imgs_perturbed = original_perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:, y_true]
    return predictions


def predict_classes_for_bit_attack(
    xs: np.ndarray, img: np.ndarray, y_true: int, model: tf.keras.Model
) -> np.ndarray:
    """Perturb the image and get the predictions of the model."""
    imgs_perturbed = original_perturb_image_by_bit_fault(xs, img)
    predictions = model.predict(imgs_perturbed)[:, y_true]
    return predictions


def attack_success(
    x: np.ndarray, img: np.ndarray, y_true: int, model: tf.keras.Model
) -> Optional[bool]:
    """Predict ONE image and return True if expected. None otherwise."""
    attack_image = original_perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    logging.debug(f"Confidence: {confidence[y_true]}")
    if predicted_class == y_true:
        return True


def attack_by_bit_success(
    x: np.ndarray, img: np.ndarray, y_true: int, model: tf.keras.Model
):
    """Predict ONE image and return True if expected. None otherwise."""
    attack_image = original_perturb_image_by_bit_fault(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    logging.debug(f"Confidence: {confidence[y_true]}")
    if predicted_class == y_true:
        return True


def build_attack(
    model: tf.keras.Model,
    pixel_count=1,
    maxiter=75,
    popsize=400,
    verbose=False,
):
    def attack(
        img: np.ndarray,
        y_true: int,
    ):
        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return predict_classes(xs, img, y_true, model)

        def callback_fn(x, convergence):
            return attack_success(
                x,
                img,
                y_true,
                model,
            )

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn,
            bounds,
            maxiter=maxiter,
            popsize=popmul,
            recombination=1,
            atol=-1,
            callback=callback_fn,
            polish=False,
        )

        if verbose:
            # Calculate some useful statistics to return from this function
            attack_image = original_perturb_image(attack_result.x, img)[0]
            prior_probs = model.predict(np.array([img]))[0]
            prior_class = np.argmax(prior_probs)
            predicted_probs = model.predict(np.array([attack_image]))[0]
            predicted_class = np.argmax(predicted_probs)
            success = predicted_class != y_true
            cdiff = prior_probs[y_true] - predicted_probs[y_true]

            print(
                dedent(
                    "-- TRUTH --\n"
                    f"y_true={y_true}\n"
                    "-- W/O FI PREDS --\n"
                    f"prior_probs={prior_probs}\n"
                    f"prior_class={prior_class}\n"
                    "-- FI PREDS --\n"
                    f"attack_results={attack_result.x}\n"
                    f"predicted_probs={predicted_probs}\n"
                    f"predicted_class={predicted_class}\n"
                    f"success={success}\n"
                    f"cdiff={cdiff}\n"
                )
            )

        return attack_result.x

    return attack


def build_attack_as_tf(
    model: tf.keras.Model,
    pixel_count=1,
    maxiter=75,
    popsize=400,
    verbose=False,
):
    attack = build_attack(
        model,
        pixel_count=pixel_count,
        maxiter=maxiter,
        popsize=popsize,
        verbose=verbose,
    )

    def attack_as_tf(img, y_true):
        return tf.numpy_function(attack, [img, y_true], tf.double)

    return attack_as_tf


def build_parallel_attack_as_tf(
    model,
    pixel_count=1,
    maxiter=75,
    popsize=400,
    verbose=False,
):
    @tf.function
    def parallel_attack_as_tf(
        imgs,
        y_trues,
    ):
        attack_as_tf = build_attack_as_tf(
            model,
            pixel_count=pixel_count,
            maxiter=maxiter,
            popsize=popsize,
            verbose=verbose,
        )
        return tf.vectorized_map(
            lambda x: attack_as_tf(x[0], x[1]), elems=[imgs, y_trues]
        )

    return parallel_attack_as_tf


def attack_by_bit(
    img: np.ndarray,
    y_true: int,
    model: tf.keras.Model,
    pixel_count=1,
    maxiter=75,
    popsize=400,
    verbose=False,
):
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes_for_bit_attack(xs, img, y_true, model)

    def callback_fn(x, convergence):
        return attack_by_bit_success(
            x,
            img,
            y_true,
            model,
        )

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn,
        bounds,
        maxiter=maxiter,
        popsize=popmul,
        recombination=1,
        atol=-1,
        callback=callback_fn,
        polish=False,
    )

    if verbose:
        # Calculate some useful statistics to return from this function
        attack_image = original_perturb_image(attack_result.x, img)[0]
        prior_probs = model.predict(np.array([img]))[0]
        prior_class = np.argmax(prior_probs)
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        success = predicted_class != y_true
        cdiff = prior_probs[y_true] - predicted_probs[y_true]

        print(
            dedent(
                "-- TRUTH --\n"
                f"y_true={y_true}\n"
                "-- W/O FI PREDS --\n"
                f"prior_probs={prior_probs}\n"
                f"prior_class={prior_class}\n"
                "-- FI PREDS --\n"
                f"attack_results={attack_result.x}\n"
                f"predicted_probs={predicted_probs}\n"
                f"predicted_class={predicted_class}\n"
                f"success={success}\n"
                f"cdiff={cdiff}\n"
            )
        )

    return attack_result.x
