import numpy as np
from matplotlib import pyplot as plt

def main():
    print("Load Data...")

    adam_itrs = np.load('./adam_itr_records.npy')
    adam_loss_records = np.load('./adam_loss_records.npy')

    gd_itrs = np.load('./gd_itr_records.npy')
    gd_loss_records = np.load('./gd_loss_records.npy')

    sgd_itrs = np.load('./sgd_itr_records.npy')
    sgd_loss_records = np.load('./sgd_loss_records.npy')

    print(adam_itrs.size)
    print(adam_loss_records.size)

    print("Visualizing...")

    fig = plt.figure(figsize=(10,6))
    axarr = fig.subplots(nrows=3,ncols=1)

    plt.subplots_adjust(wspace= 0, hspace=0.6)
 
    axarr[0].title.set_text('Adam Plot')
    axarr[0].plot(adam_itrs, adam_loss_records, label='Training Loss')

    axarr[1].title.set_text('SGD Plot')
    axarr[1].plot(sgd_itrs, sgd_loss_records, label='Training Loss')

    axarr[2].title.set_text('GD Plot')
    axarr[2].plot(gd_itrs, gd_loss_records, label='Training Loss')

    plt.savefig('./loss-graphic.png', orientation='landscape')
    plt.show()


if __name__ == '__main__':
    main()

