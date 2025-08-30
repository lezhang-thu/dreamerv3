import os
import threading
import collections

import elements
import embodied
import numpy as np
import gymnasium as gym
import ale_py

from PIL import Image


class AtariX(embodied.Env):

    LOCK = threading.Lock()
    WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

    def __init__(
        self,
        name,
        repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives='unused',
        sticky=True,
        actions='all',
        length=108000,
        pooling=2,
        aggregate='max',
        resize='pillow',
        autostart=False,
        clip_reward=False,
        seed=None,
    ):

        assert lives in ('unused', 'discount', 'reset'), lives
        assert actions in ('all', 'needed'), actions
        assert resize in ('opencv', 'pillow'), resize
        assert aggregate in ('max', 'mean'), aggregate
        assert pooling >= 1, pooling
        assert repeat >= 1, repeat
        if name == 'james_bond':
            name = 'jamesbond'
        self.name = "".join(word.capitalize() for word in name.split("_")) 

        self.repeat = repeat
        self.size = size
        self.gray = gray
        self.length = length
        self.pooling = pooling
        self.aggregate = aggregate
        self.resize = resize
        self.autostart = autostart
        self.clip_reward = clip_reward

        with self.LOCK:
            self._env = gym.make(
                'ALE/{}-v5'.format(self.name),
                obs_type='rgb',  # ram | rgb | grayscale
                frameskip=1,  # frame skip
                mode=None,  # game mode, see Machado et al. 2018
                difficulty=None,  # game difficulty, see Machado et al. 2018
                repeat_action_probability=0.0,  # Sticky action probability
                full_action_space=actions == "all",  # Use all actions
                render_mode=None  # None | human | rgb_array
            )
        self.ale = self._env.unwrapped.ale
        self.actionset = {
            'all': self.ale.getLegalActionSet,
            'needed': self.ale.getMinimalActionSet,
        }[actions]()

        W, H = self.ale.getScreenDims()
        self.buffers = collections.deque(
            [np.zeros((W, H, 3), np.uint8) for _ in range(self.pooling)],
            maxlen=self.pooling)
        self.prevlives = None
        self.duration = None
        self.done = True

    @property
    def obs_space(self):
        return {
            'image': elements.Space(np.uint8,
                                    (*self.size, 1 if self.gray else 3)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, len(self.actionset)),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self.done:
            self._reset(0)
            self.prevlives = self.ale.lives()
            self.duration = 0
            self.done = False
            return self._obs(0.0, is_first=True)
        reward = 0.0
        terminal = False
        last = False
        assert 0 <= action['action'] < len(self.actionset), action['action']
        act = self.actionset[action['action']]
        for repeat in range(self.repeat):
            reward += self.ale.act(act)
            self.duration += 1
            if repeat >= self.repeat - self.pooling:
                self._render()
            if self.ale.game_over():
                terminal = True
                last = True
            if self.duration >= self.length:
                last = True
            lives = self.ale.lives()
            self.prevlives = lives
            if terminal or last:
                break
        self.done = last
        obs = self._obs(reward, is_last=last, is_terminal=terminal)
        return obs

    def _reset(self, seed=None):
        self._env.reset(seed=seed)
        self.ale = self._env.unwrapped.ale
        self._render()
        for i, dst in enumerate(self.buffers):
            if i > 0:
                np.copyto(self.buffers[0], dst)

    def _render(self, reset=False):
        self.buffers.appendleft(self.buffers.pop())
        self.ale.getScreenRGB(self.buffers[0])

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        if self.clip_reward:
            reward = np.sign(reward)
        if self.aggregate == 'max':
            image = np.amax(self.buffers, 0)
        elif self.aggregate == 'mean':
            image = np.mean(self.buffers, 0).astype(np.uint8)
        if self.resize == 'opencv':
            import cv2
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        elif self.resize == 'pillow':
            image = Image.fromarray(image)
            image = image.resize(self.size, Image.BILINEAR)
            image = np.array(image)
        if self.gray:
            # Averaging channels equally would not work. For example, a fully red
            # object on a fully green background would average to the same color.
            image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)[:, :,
                                                                       None]
        return dict(
            image=image,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_last,
        )
