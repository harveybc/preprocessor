import sys
import preprocessor.feature_selector.feature_selector as dt


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]
    # you want your project to do.
    dt.run(args)


if __name__ == "__main__":
    main()
