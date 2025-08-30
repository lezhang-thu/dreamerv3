import random

import numpy as np


class Replay:

    def __init__(self, batch_size, batch_length):
        self.batch_size = batch_size
        self.batch_length = batch_length

        self._storage = []
        self._maxsize = int(1e3)
        self._next_idx = 0

        self.worker_traj = dict()

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return self

    def uniform_traj(self):
        batch_size = self.batch_size
        batch_length = self.batch_length
        keys_set = self._storage[0][0].keys()

        while True:
            trajectories = random.sample(self._storage, k=batch_size)
            for i, x in enumerate(trajectories):
                x_start = random.randrange(len(x) + 1)  # inclusive 0..len(x)
                trajectories[i] = x[x_start:]

            start = 0
            dict_list = []
            segment_lengths = []

            for traj in trajectories:
                traj_len = len(traj)
                segment_dict = {
                    key:
                    np.stack([
                        traj_e[key]
                        for traj_e in traj[start:start + batch_length]
                    ], 0) if start < traj_len else None
                    for key in keys_set
                }

                dict_list.append(segment_dict)
                segment_len = min(start + batch_length,
                                  traj_len) - start if start < traj_len else 0
                segment_lengths.append(segment_len)
            result = {}
            for key in dict_list[0].keys():
                values = [d[key] for d in dict_list]
                arrays = [v for v in values if v is not None]
                shapes = [arr.shape for arr in arrays]
                assert all(shape[1:] == shapes[0][1:] for shape in shapes)
                max_d0 = batch_length
                common_shape_tail = shapes[0][1:]
                common_dtype = np.result_type(*arrays)
                padded_arrays = []
                for value in values:
                    arr = (np.array(
                        [], dtype=common_dtype).reshape((0, ) +
                                                        common_shape_tail)
                           if value is None else value)
                    if arr.shape[0] < max_d0:
                        pad_width = [(0, max_d0 - arr.shape[0])
                                     ] + [(0, 0)] * (arr.ndim - 1)
                        arr = np.pad(arr,
                                     pad_width,
                                     mode='constant',
                                     constant_values=0)
                    padded_arrays.append(arr)
                result[key] = np.stack(padded_arrays, 0)
            result['is_first'][:, 0] = True
            mask = np.zeros((batch_size, batch_length), dtype=bool)
            for x_k, length in enumerate(segment_lengths):
                mask[x_k, :length] = True
            result['mask'] = mask
            result['last_chunk'] = True
            yield result

    def generator(self):
        batch_size = self.batch_size
        batch_length = self.batch_length
        keys_set = self._storage[0][0].keys()
        while True:
            trajectories = random.sample(self._storage, k=batch_size)
            max_len = max(len(t) for t in trajectories)

            num_chunks = -(-max_len // batch_length)
            for k in range(num_chunks):
                start = k * batch_length
                dict_list = []
                segment_lengths = []

                for traj in trajectories:
                    traj_len = len(traj)
                    segment_dict = {
                        key:
                        np.stack([
                            traj_e[key]
                            for traj_e in traj[start:start + batch_length]
                        ], 0) if start < traj_len else None
                        for key in keys_set
                    }

                    dict_list.append(segment_dict)
                    segment_len = min(start + batch_length, traj_len
                                      ) - start if start < traj_len else 0
                    segment_lengths.append(segment_len)
                result = {}
                for key in dict_list[0].keys():
                    values = [d[key] for d in dict_list]
                    arrays = [v for v in values if v is not None]
                    shapes = [arr.shape for arr in arrays]
                    assert all(shape[1:] == shapes[0][1:] for shape in shapes)
                    max_d0 = batch_length
                    common_shape_tail = shapes[0][1:]
                    common_dtype = np.result_type(*arrays)
                    padded_arrays = []
                    for value in values:
                        arr = (np.array(
                            [], dtype=common_dtype).reshape((0, ) +
                                                            common_shape_tail)
                               if value is None else value)
                        if arr.shape[0] < max_d0:
                            pad_width = [(0, max_d0 - arr.shape[0])
                                         ] + [(0, 0)] * (arr.ndim - 1)
                            arr = np.pad(arr,
                                         pad_width,
                                         mode='constant',
                                         constant_values=0)
                        padded_arrays.append(arr)
                    result[key] = np.stack(padded_arrays, 0)
                mask = np.zeros((batch_size, batch_length), dtype=bool)
                for x_k, length in enumerate(segment_lengths):
                    mask[x_k, :length] = True
                result['mask'] = mask
                result['last_chunk'] = k == num_chunks - 1
                yield result

    def add(self, step, worker=0):
        step = {k: v for k, v in step.items() if not k.startswith('log/')}
        if worker not in self.worker_traj:
            self.worker_traj[worker] = list()

        self.worker_traj[worker].append(step)
        if step['is_last']:
            # finish the current traj., so move to the next new one
            self._to_storage(self.worker_traj[worker])
            self.worker_traj[worker] = list()

    def _to_storage(self, traj):
        if self._next_idx >= len(self._storage):
            self._storage.append(traj)
        else:
            self._storage[self._next_idx] = traj
        self._next_idx = (self._next_idx + 1) % self._maxsize
