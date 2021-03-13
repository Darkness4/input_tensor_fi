from textwrap import dedent
from typing import List, Optional

import numpy as np
import tensorflow as tf
from inputtensorfi.attacks.differential_evolution import differential_evolution
from inputtensorfi.helpers import utils
from inputtensorfi.manipulation.img.utils import original_perturb_image


def predict_classes(
    xs: np.ndarray, img: np.ndarray, y_true: int, model: tf.keras.Model
) -> np.ndarray:
    """Perturb the image and get the predictions of the model.

    Args:
        xs (np.ndarray): A 2D array of PixelFault.
        img (np.ndarray): One image (shape=(h, v, channels)).
        y_true (int): Index of the true class (categorical).
        model (tf.keras.Model): A model (not faulted).
        minimize (bool, optional): Return the complement if needed.
            Defaults to True.

    Returns:
        np.ndarray: predictions for the true model, i.e.:
            model.predict(imgs)[:, y_true]
    """
    imgs_perturbed = original_perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:, y_true]
    return predictions


def attack_success(
    x: np.ndarray,
    img: np.ndarray,
    y_true: int,
    model: tf.keras.Model,
    verbose=False,
) -> Optional[bool]:
    """Predict ONE image and return True if expected. None otherwise."""
    attack_image = original_perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print("Confidence:", confidence[y_true])
    if predicted_class == y_true:
        return True


def attack(
    img: np.ndarray,
    y_true: int,
    model: tf.keras.Model,
    class_names: List[str],
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
        return predict_classes(xs, img, y_true, model)

    def callback_fn(x, convergence):
        return attack_success(
            x,
            img,
            y_true,
            model,
            verbose,
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

    # Calculate some useful statistics to return from this function
    attack_image = original_perturb_image(attack_result.x, img)[0]
    prior_probs = model.predict(np.array([img]))[0]
    prior_class = np.argmax(prior_probs)
    predicted_probs = model.predict(np.array([attack_image]))[0]
    predicted_class = np.argmax(predicted_probs)
    success = predicted_class != y_true
    cdiff = prior_probs[y_true] - predicted_probs[y_true]

    if verbose:
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
