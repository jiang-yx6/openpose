import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pose.settings')
django.setup()

from daphne.cli import CommandLineInterface

if __name__ == '__main__':
    cli = CommandLineInterface()
    cli.run(['pose.asgi:application'])