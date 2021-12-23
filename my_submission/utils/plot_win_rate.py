import matplotlib.pyplot as plt

def plot_win_rate(win_rate, name='1.jpg'):
    print('win_rate:{}'.format(win_rate))
    x = [i for i in range(len(win_rate))]
    y= [win_rate[0]]
    for i in range(1,len(win_rate)):
        avg = sum(win_rate[:i+1])/(i+1)
        y.append(avg)
    plt.plot(x, y, color='r',linestyle='dashed')
    plt.ylim((0, 1))
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.savefig('{}.jpg'.format(name),dpi=200, bbox_inches = 'tight')
    print('plot win_rate is finished')