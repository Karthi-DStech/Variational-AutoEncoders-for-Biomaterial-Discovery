import os
import sys
import time

from call_methods import make_dataset, make_model
from options.train_options import TrainOptions
from utils import tb_visualizer
from utils.utils import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run():
    """
    Run the training process

    Parameters
    ----------
    None

    Returns
    -------
    None

    Process
    -------
    1. Parse the training options
    2. Set the random seed
    3. Create the model
    4. Create the dataset
    5. Create the visualizer
    6. Train the model
    7. Save the model
    """

    opt = TrainOptions().parse()

    set_seed(opt.seed)

    model = make_model(opt.model_name, opt)
    train_dataset = make_dataset(dataset_name=opt.dataset_name, opt=opt)

    visualizer = tb_visualizer.Visualizer(opt)
    start = time.time()

    epoch = 0
    for epoch in range(opt.n_epochs):
        train_epoch_start = time.time()
        for i, data in enumerate(train_dataset.dataloader):
            total_steps = epoch * len(train_dataset.dataloader) + i
            do_visualization = total_steps % opt.save_image_frequency == 0
            model.set_input(data)
            model.train(do_visualization=do_visualization)
            visualizer.log_performance(
                model.performance,
                epoch=epoch,
                step=i,
                total_steps=total_steps,
                is_train=True,
                print_freq=opt.print_freq,
            )
            if do_visualization:
                visualizer.log_images(
                    model.vis_data, total_steps=total_steps, is_train=True
                )
        train_epoch_end = time.time()
        visualizer.log_time(train_epoch_end, train_epoch_start, epoch, is_train=True)

        if epoch % opt.model_save_frequency == 0 or epoch == opt.n_epochs - 1:
            model.save_networks(visualizer.log_dir, epoch)
    end = time.time()
    visualizer.log_time(end, start, epoch, training_end=True)
    visualizer.close()


if __name__ == "__main__":
    run()
