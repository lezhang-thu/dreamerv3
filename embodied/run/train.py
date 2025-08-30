import collections
from functools import partial as bind
import pickle
import copy

import elements
import embodied
import numpy as np


def _ge_replay(
    make_ge_env,
    make_replay,
):
    env = make_ge_env()
    with open("{}.pkl".format(env.name), 'rb') as f:
        action_seqs = pickle.load(f)
    ge_replay = make_replay()

    for item in action_seqs:
        act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
        act['reset'] = True
        list_of_actions, cumulative_reward, timestamp = item
        timestamp = 'go-explore-{}'.format(timestamp)
        score = 0

        obs = env.step(act)
        obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
        score += obs['reward']
        for a in list_of_actions:
            act = {
                k: np.full(v.shape, a, v.dtype)
                for k, v in env.act_space.items()
            }
            act['reset'] = False

            t_act = copy.deepcopy(act)
            t_act.pop('reset')
            trans = {**obs, **t_act}
            ge_replay.add(trans)

            obs = env.step(act)
            obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
            score += obs['reward']
            if obs['is_last']:
                act = {
                    k: np.zeros(v.shape, v.dtype)
                    for k, v in env.act_space.items()
                }
                act['reset'] = True

                t_act = copy.deepcopy(act)
                t_act.pop('reset')
                trans = {**obs, **t_act}
                ge_replay.add(trans)
        assert score == cumulative_reward, 'score: {} vs. required: {}'.format(
            score, cumulative_reward)
        print('t: {: >20}, score: {: >8}, #transitions: {: >5}'.format(
            timestamp, score, len(list_of_actions)))
        #if score > 10_000:
        #    break
    return ge_replay


def train(
    make_agent,
    make_replay,
    make_env,
    make_ge_env,
    make_logger,
    args,
):

    agent = make_agent()
    replay = make_replay()
    ge_replay = _ge_replay(make_ge_env, make_replay)
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    #should_report = embodied.LocalClock(args.report_every)
    should_save = embodied.LocalClock(args.save_every)

    @elements.timer.section('logfn')
    def logfn(tran, worker):
        episode = episodes[worker]
        tran['is_first'] and episode.reset()
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')
        for key, value in tran.items():
            if value.dtype == np.uint8 and value.ndim == 3:
                if worker == 0:
                    episode.add(f'policy_{key}', value, agg='stack')
            elif key.startswith('log/'):
                assert value.ndim == 0, (key, value.shape, value.dtype)
                episode.add(key + '/avg', value, agg='avg')
                episode.add(key + '/max', value, agg='max')
                episode.add(key + '/sum', value, agg='sum')
        if tran['is_last']:
            result = episode.result()
            logger.add(
                {
                    'score': result.pop('score'),
                    'length': result.pop('length'),
                },
                prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1])
                                         >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.envs)]
    driver = embodied.Driver(fns, parallel=not args.debug)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    carry_train = [agent.init_train(args.batch_size)]
    #carry_report = agent.init_report(args.batch_size)

    #replay_gen = replay.generator()
    #ge_replay_gen = ge_replay.generator()
    replay_gen = replay.uniform_traj()
    ge_replay_gen = ge_replay.uniform_traj()
    state = 0

    def trainfn(tran, worker):
        nonlocal state
        if len(replay) < args.batch_size:
            return
        for _ in range(should_train(step)):
            t_gen = replay_gen if state == 0 else ge_replay_gen
            x_get = next(t_gen)
            if x_get.pop("last_chunk"):
                state = 1 - state
            batch = agent.stream(x_get)
            carry_train[0], outs, mets = agent.train(carry_train[0], batch)
            train_fps.step(batch_steps)
            train_agg.add(mets, prefix='train')

    driver.on_step(trainfn)

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    if args.from_checkpoint:
        elements.checkpoint.load(
            args.from_checkpoint,
            dict(agent=bind(agent.load, regex=args.from_checkpoint_regex)))
    cp.load_or_save()

    print('Start training loop')
    policy = lambda *args: agent.policy(*args, mode='train')
    driver.reset(agent.init_policy)
    while step < args.steps:

        driver(policy, steps=10)

        #if should_report(step):
        #  agg = elements.Agg()
        #  logger.add(agg.result(), prefix='report')

        if should_log(step):
            logger.add(train_agg.result())
            logger.add(epstats.result(), prefix='epstats')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.add({'timer': elements.timer.stats()['summary']})
            logger.write()

        if should_save(step):
            cp.save()

    logger.close()
