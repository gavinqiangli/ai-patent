import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap


# Define block positions and sizes
blocks = {
    'Block 1 Block 1 Block 1 Block 1': (0, 0, 1, 1),
    'Block 2': (0, 2, 1, 1),
    'Block 3': (2, 0, 1, 1),
    'Block 4': (2, 2, 1, 1)
}

# Define connections between blocks
connections = {
    "Block 1 Block 1 Block 1 Block 1": ["Block 2", "Block 4"],
    "Block 2": ["Block 3", "Block 4"],
    "Block 3": ["Block 1 Block 1 Block 1 Block 1"],
    "Block 4": ["Block 3"]
}

# Function to draw blocks
def draw_blocks(blocks):
    for block, (x, y, width, height) in blocks.items():
        plt.gca().add_patch(Rectangle((x, y), width, height, fill=True, edgecolor='black', facecolor='lightgrey'))
        wrapped_text = '\n'.join(textwrap.wrap(block, width=10))  # Adjust width as needed
        plt.text(x + width / 2, y + height / 2, wrapped_text, ha='center', va='center', fontsize=10)

# Function to draw arrows
def draw_arrows(connections, blocks):
    for block, destinations in connections.items():
        for dest_block in destinations:
            x_start, y_start, _, _ = blocks[block]
            x_end, y_end, _, _ = blocks[dest_block]
            plt.arrow(x_start + 0.5, y_start + 0.5, x_end - x_start, y_end - y_start,
                      head_width=0.01, head_length=0.01, fc='lightgrey', ec='lightgrey')

# Set up plot
plt.figure(figsize=(6, 4))
plt.axis('off')

# Draw blocks and arrows
draw_blocks(blocks)
draw_arrows(connections, blocks)

# Add figure name
plt.text(2, 4, 'Figure A: Block Diagram', fontsize=12, ha='center')

# Show plot
plt.show()
