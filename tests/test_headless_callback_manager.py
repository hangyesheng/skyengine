"""Regression tests for headless callback construction.

Headless benchmark workers must not initialize Pygame.  On macOS, leaving the
SDL joystick subsystem to Pygame's interpreter-exit hook can terminate an
otherwise successful worker with SIGTRAP (exit code 133).
"""

import pygame

from executor.packet_factory.call_back.callback_manager.CallbackManager import CallbackManager


def test_headless_callback_manager_does_not_initialize_pygame():
    pygame.quit()
    manager = CallbackManager(include_visualizer=False)

    assert not manager.has("initialize_visualizer")
    assert not pygame.get_init()


def test_visual_callback_manager_still_registers_visualizer():
    pygame.quit()
    manager = CallbackManager(include_visualizer=True)

    try:
        assert manager.has("initialize_visualizer")
        assert pygame.get_init()
    finally:
        pygame.quit()
