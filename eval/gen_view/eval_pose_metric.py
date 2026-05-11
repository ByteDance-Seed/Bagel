import matplotlib.pyplot as plt

from gen_view_eval.cli import main


if __name__ == '__main__':
    plt.switch_backend("agg")
    main()
