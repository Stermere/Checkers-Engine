# file to visualize the performance of the model

import matplotlib.pyplot as plt

def error_plot():
    # open the error.txt file and read the contents in to a list of tuples
    error = []
    index = []
    with open('error.txt', 'r') as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            index.append(i)
            error.append(float(line))
            i += 1
    # avg the error for every 5 epochs
    error_avg5 = []
    index5 = []
    for i in range(0, len(error), 10):
        error_avg5.append(sum(error[i:i+10])/10)
        index5.append(i)

    # plot the data
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    # add a title to the subplots
    ax[0].set_title('Error vs Epoch')
    ax[1].set_title('Error vs Epochs (avg 10 epochs)')
    # adjust the plots to show the top title
    fig.subplots_adjust(top=0.95)

    ax[0].scatter(index, error, linewidth=2.0)
    ax[1].scatter(index5, error_avg5, linewidth=2.0)

    plt.show()

    

    






def main():
    error_plot()


if __name__ == '__main__':
    main()
