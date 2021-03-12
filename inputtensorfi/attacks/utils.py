def attack_success(
    x, img, target_class: int, model, targeted_attack=False, verbose=False
):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print("Confidence:", confidence[target_class])
    if (targeted_attack and predicted_class == target_class) or (
        not targeted_attack and predicted_class != target_class
    ):
        return True


def attack(
    img_id: int,
    model,
    target=None,
    pixel_count=1,
    maxiter=75,
    popsize=400,
    verbose=False,
):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img_id, 0]

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(
            xs, x_test[img_id], target_class, model, target is None
        )

    def callback_fn(x, convergence):
        return attack_success(
            x, x_test[img_id], target_class, model, targeted_attack, verbose
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
    attack_image = perturb_image(attack_result.x, x_test[img_id])[0]
    prior_probs = model.predict_one(x_test[img_id])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    return [
        model.name,
        pixel_count,
        img_id,
        actual_class,
        predicted_class,
        success,
        cdiff,
        prior_probs,
        predicted_probs,
        attack_result.x,
    ]
