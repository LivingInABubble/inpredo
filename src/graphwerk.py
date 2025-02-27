import os

import matplotlib.pyplot as plt
import numpy as np
from mplfinance.original_flavor import candlestick2_ochl
from tqdm import tqdm


def convolve_sma(array, period):
    return np.convolve(array, np.ones((period,)) / period, mode='valid')


def graphwerk(start, finish):
    opened, closed, high, low, volume, date = [], [], [], [], [], []
    for x in range(start, finish):
        # Below filtering is valid for eurusd.csv file. Other financial data files have different orders
        # so you need to find out what means open, high and closed in their respective order.
        opened.append(float(pd[x][3]))
        high.append(float(pd[x][4]))
        low.append(float(pd[x][5]))
        closed.append(float(pd[x][6]))
        volume.append(float(pd[x][7]))
        date.append(pd[x][1])

    close_next = float(pd[finish][6])

    sma = convolve_sma(closed, 5)
    smb = list(sma)
    diff = sma[-1] - sma[-2]

    for x in range(len(closed) - len(smb)):
        smb.append(smb[-1] + diff)

    fig = plt.figure(num=1, figsize=(3, 3), dpi=50, facecolor='w', edgecolor='k')
    dx = fig.add_subplot(111)
    # mpl_finance.volume_overlay(ax, opened, closed, volume, width=0.4, colorup='b', colordown='b', alpha=1)
    candlestick2_ochl(dx, opened, closed, high, low, width=1.5, colorup='g', colordown='r', alpha=0.5)

    plt.autoscale()
    plt.plot(smb, color="blue", linewidth=10, alpha=0.5)
    plt.axis('off')
    # comp_ratio = close_next / closed[-1]
    # print(comp_ratio)

    filename = f'{date[0]} ~ {date[-1]}.jpg'.replace(':00:00', '')
    if closed[-1] > close_next:
        # print('close value is bigger')
        # print('last value: ' + str(closed[-1]))
        # print('next value: ' + str(close_next))
        # print('sell')
        plt.savefig(sell_dir + filename, bbox_inches='tight')
    else:
        # print('close value is smaller')
        # print('last value: ' + str(closed[-1]))
        # print('next value: ' + str(close_next))
        # print('buy')
        plt.savefig(buy_dir + filename, bbox_inches='tight')

    # plt.show()
    plt.close()


if __name__ == '__main__':
    buy_dir = '../data/buy/'
    if not os.path.exists(buy_dir):
        os.makedirs(buy_dir)

    sell_dir = '../data/sell/'
    if not os.path.exists(sell_dir):
        os.makedirs(sell_dir)

    # Input your csv file here with historical data
    ad = np.genfromtxt('../financial_data/gemini_BTCUSD_1hr.csv', delimiter=',', dtype=str)
    pd = np.flipud(ad)

    for i in tqdm(range(int(len(pd) / 2) - 6)):
        graphwerk(i * 2, i * 2 + 12)
