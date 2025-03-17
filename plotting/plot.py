import matplotlib.pyplot as plt


def iter_dataloader(dataloader):
    return next(iter(dataloader))


def getting_number_rows(images):
    return 3 if len(images) > 3 else len(images)


def getting_fig_size(number_rows):
    return (10, 10) if number_rows == 3 else (10, 5)


def show_sample_images(dataloader):
    images, masks = iter_dataloader(dataloader)
    number_rows = getting_number_rows(images)
    fig_size = getting_fig_size(number_rows)
    fig, (grid) = plt.subplots(number_rows, 3, figsize=fig_size)
    fig.suptitle('Sample images', fontsize=20)
    for i in range(number_rows):
        grid[i][0].imshow(images[i].permute(1, 2, 0), cmap='gray')
        grid[i][0].set_axis_off()
        grid[i][0].set_title(f'Image')

        grid[i][1].imshow(masks[i].permute(1, 2, 0), cmap='gray')
        grid[i][1].set_axis_off()
        grid[i][1].set_title(f'Mask')

        grid[i][2].imshow(images[i].permute(1, 2, 0), cmap='gray')
        grid[i][2].imshow(masks[i].permute(1, 2, 0), cmap="gray", alpha=0.5)
        grid[i][2].set_axis_off()
        grid[i][2].set_title(f'Mask')
    plt.show()


def graph_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    