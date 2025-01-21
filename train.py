import constellation
from constellation import util
import torch
from matplotlib import pyplot
from mpl_toolkits.axisartist.axislines import SubplotZero
import warnings

torch.manual_seed(57)

# Number of symbols to learn
order = 16

# Shape of the hidden layers
hidden_layers = (8, 4,)

# Initial value for the learning rate
initial_learning_rate = 0.1

# Number of batches to skip between every loss report
loss_report_batch_skip = 50

# Size of batches
batch_size = 2048

# File in which the trained model is saved
output_file = 'output/constellation-order-{}.pth'.format(order)

###

# Setup plot for showing training progress
fig = pyplot.figure()
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

pyplot.show(block=False)

# Train the model with random data
model = constellation.ConstellationNet(
    order=order,
    encoder_layers=hidden_layers,
    decoder_layers=hidden_layers[::-1],
)

print('Starting training\n')

# Current batch index
batch = 0

# Accumulated loss for last batches
running_loss = 0

# List of training examples (not shuffled)
classes_ordered = torch.arange(order).repeat(batch_size)

# Constellation from the previous training batch
prev_constel = model.get_constellation()
total_change = float('inf')

# Optimizer settings
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, verbose=True,
    factor=0.25,
    patience=100,
    cooldown=50,
    threshold=1e-8
)

while total_change >= 1e-3:
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

    # Report loss
    running_loss += loss.item()

    if batch % loss_report_batch_skip == loss_report_batch_skip - 1:
        print('Batch #{}'.format(batch + 1))
        print('\tLoss is {}'.format(running_loss / loss_report_batch_skip))
        print('\tChange is {}\n'.format(total_change))

        running_loss = 0

    # Update figure with current encoding
    ax.clear()
    util.plot_constellation(
        ax, cur_constel,
        model.channel, model.decoder,
        noise_samples=0
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    batch += 1

model.eval()

# Calcul de la perte finale
with torch.no_grad():
    classes_ordered = torch.arange(order).repeat(2048)
    classes_dataset = classes_ordered[torch.randperm(len(classes_ordered))]
    onehot_dataset = util.messages_to_onehot(classes_dataset, order)

    predictions = model(onehot_dataset)
    final_loss = criterion(predictions, classes_dataset)

print('\nFinished training')
print('Final loss is {}'.format(final_loss))
print('Saving model as {}'.format(output_file))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    torch.save(model, output_file)
