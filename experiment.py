import constellation
from constellation import util
import math
import torch
import time
import pickle
import random
import sys

# Number of seconds to wait between each checkpoint
time_between_checkpoints = 10 * 60  # 10 minutes

# Format for checkpoint files
checkpoint_path = 'output/experiment-{}.pkl'


def train_with_parameters(
    order,
    layer_sizes,
    initial_learning_rate,
    batch_size
):
    """
    Report final loss after fully learning a constellation with given
    parameters.

    :param order: Number of symbols in the constellation.
    :param layer_sizes: Shape of the encoder’s hidden layers. The
    size of this sequence is the number of hidden layers, with each element
    being a number which specifies the number of neurons in its channel. The
    decoder’s hidden layers will be of the same shape but reversed.
    :param initial_learning_rate: Initial learning rate used for the optimizer.
    :param batch_size: Number of training examples for each training batch
    expressed as a multiple of the constellation order.
    """
    model = constellation.ConstellationNet(
        order=order,
        encoder_layers=layer_sizes,
        decoder_layers=layer_sizes[::-1],
    )

    # List of training examples (not shuffled)
    classes_ordered = torch.arange(order).repeat(batch_size)

    # Constellation from the previous training batch
    prev_constel = model.get_constellation()
    total_change = float('inf')

    # Optimizer settings
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.25,
        patience=100,
        cooldown=50,
        threshold=1e-8
    )

    while total_change >= 1e-4:
        # Shuffle training data and convert to one-hot encoding
        classes_dataset = classes_ordered[torch.randperm(len(classes_ordered))]
        onehot_dataset = util.messages_to_onehot(classes_dataset, order)

        # Perform training step for current batch
        model.train()
        optimizer.zero_grad()
        predictions = model(onehot_dataset)
        loss = criterion(predictions, classes_dataset)
        loss.backward()
        optimizer.step()

        # Update learning rate scheduler
        scheduler.step(loss)

        # Check for convergence
        model.eval()
        cur_constel = model.get_constellation()
        total_change = (cur_constel - prev_constel).norm(dim=1).sum()
        prev_constel = cur_constel

    # Compute final loss value
    with torch.no_grad():
        classes_ordered = torch.arange(order).repeat(2048)
        classes_dataset = classes_ordered[torch.randperm(len(classes_ordered))]
        onehot_dataset = util.messages_to_onehot(classes_dataset, order)

        predictions = model(onehot_dataset)
        return criterion(predictions, classes_dataset).tolist()


def evaluate_parameters(parameters, num_repeats=3):
    """
    Run constellation training several times and keep the lowest reached loss.

    :param parameters: Training parameters (see `train_with_parameters` for
    documentation).
    :param num_repeats: Number of runs.
    :return: Lowest reached loss.
    """
    minimal_loss = float('inf')

    for run_index in range(num_repeats):
        current_loss = train_with_parameters(**parameters)
        minimal_loss = min(minimal_loss, current_loss)

    return minimal_loss


def generate_test_configurations():
    """
    Generate the set of all configurations to be tested.

    :yield: Configuration as a dictionary of parameters.
    """
    # Cartesian product of independent variables
    independent_vars = util.product_dict(
        order=[4, 16, 32],
        initial_learning_rate=[10 ** x for x in range(-2, 1)],
        batch_size=[8, 2048],
    )

    # Add dependent variables
    for current_dict in independent_vars:
        for first_layer in range(0, current_dict['order'] + 1, 4):
            for last_layer in range(0, first_layer + 1, 4):
                # Convert pair of sizes for each layer to a shape tuple
                if first_layer == 0 and last_layer == 0:
                    layer_sizes = ()
                elif first_layer != 0 and last_layer == 0:
                    layer_sizes = (first_layer,)
                elif first_layer == 0 and last_layer != 0:
                    layer_sizes = (last_layer,)
                else:  # first_layer != 0 and last_layer != 0
                    layer_sizes = (first_layer, last_layer)

                # Merge dependent variables with independent ones
                yield {
                    **current_dict,
                    'layer_sizes': layer_sizes
                }


def save_results(results, path):
    """
    Save current results of experiment.

    :param results: Dictionary containing current results.
    :param path: Path to the file where results are to be saved.
    """
    with open(path, 'wb') as file:
        pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)


# List of all configurations to be tested
random.seed(42)
all_confs = list(generate_test_configurations())
random.shuffle(all_confs)

# Number of splits of the configuration list
parts_count = 1

# Current split of the configuration list
current_part = 0

if len(sys.argv) == 2:
    print('Please specify which part must be evaluated.', file=sys.stderr)
    sys.exit(1)

if len(sys.argv) == 3:
    parts_count = int(sys.argv[1])
    current_part = int(sys.argv[2]) - 1

    if parts_count < 1:
        print('There must be at least one part.', file=sys.stderr)
        sys.exit(1)

    if current_part < 0 or current_part >= parts_count:
        print(
            'Current part must be between 1 and the number of parts.',
            file=sys.stderr
        )
        sys.exit(1)

# Starting/ending index of configurations to be tested
part_start_index = math.floor(current_part * len(all_confs) / parts_count)
part_end_index = math.floor((current_part + 1) * len(all_confs) / parts_count)
part_size = part_end_index - part_start_index

if parts_count == 1:
    print('Evaluating the whole set of configurations')
    print('Use “{} [parts_count] [current_part]” to divide it'.format(
        sys.argv[0]
    ))
else:
    print('Evaluating part {}/{} of the set of configurations'.format(
        current_part,
        parts_count
    ))
    print('(indices {} to {})'.format(part_start_index, part_end_index - 1))

print()

# Current set of results
results = {}

# Last checkpoint save time
last_save_time = 0

for conf in all_confs[part_start_index:part_end_index]:
    key = tuple(sorted(conf.items()))
    results[key] = evaluate_parameters(conf)

    print('{}/{} configurations tested ({:.1f} %)'.format(
        len(results), part_size,
        100 * len(results) / part_size,
    ))

    current_time = math.floor(time.time())

    if current_time - last_save_time >= time_between_checkpoints:
        current_path = checkpoint_path.format(current_time)
        save_results(results, current_path)
        print('Saved checkpoint to {}'.format(current_path))
        last_save_time = current_time

# Save final checkpoint
output_path = checkpoint_path.format('final')
save_results(results, output_path)
print('Saved results to {}'.format(output_path))
