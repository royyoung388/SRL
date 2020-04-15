import matplotlib.pyplot as plt
import numpy as np

from optimizer import NoamOpt

if __name__ == '__main__':
    opts = [NoamOpt(200, 1, 400, None),
            NoamOpt(200, 1, 800, None),
            NoamOpt(100, 1, 400, None)]
    plt.subplot(221)
    plt.plot(np.arange(1, 2000), [[opt.rate(i) for opt in opts] for i in range(1, 2000)])
    plt.legend(["200:1:400", "200:1:800", "100:1:400"])

    opts = [NoamOpt(200, 4, 400, None),
            NoamOpt(200, 4, 800, None),
            NoamOpt(100, 4, 400, None)]
    plt.subplot(222)
    plt.plot(np.arange(1, 2000), [[opt.rate(i) for opt in opts] for i in range(1, 2000)])
    plt.legend(["200:4:400", "200:4:800", "100:4:400"])

    opts = [NoamOpt(300, 1, 400, None),
            NoamOpt(300, 1, 800, None),
            NoamOpt(200, 1, 400, None)]
    plt.subplot(223)
    plt.plot(np.arange(1, 2000), [[opt.rate(i) for opt in opts] for i in range(1, 2000)])
    plt.legend(["300:1:400", "300:1:800", "200:1:400"])

    opts = [NoamOpt(300, 4, 400, None),
            NoamOpt(300, 4, 800, None),
            NoamOpt(200, 4, 400, None)]
    plt.subplot(224)
    plt.plot(np.arange(1, 2000), [[opt.rate(i) for opt in opts] for i in range(1, 2000)])
    plt.legend(["300:4:400", "300:4:800", "200:4:400"])

    plt.show()