"""Run the six-method bottle comparison using the resumable suite."""
import sys

from vdd.suite import main

if __name__ == '__main__':
    if '--categories' not in sys.argv:
        sys.argv.extend(['--categories', 'bottle'])
    main()
