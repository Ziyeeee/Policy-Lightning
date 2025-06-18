from typing import Optional
import numpy as np
import numba


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, sequence_length: int, 
    episode_mask: np.ndarray,
    pad_before: int = 0, pad_after: int = 0,
    debug: bool = True) -> np.ndarray:
    """
    创建采样序列的索引。该函数为每个 episode 创建连续的采样窗口（即一个个子序列） 
    对应回放缓冲区中的数据。

    :param episode_ends: 一个包含每个 episode 结束位置的数组
    :param sequence_length: 每个采样序列的长度
    :param episode_mask: 一个布尔数组，标记哪些 episode 被用于训练（True 表示该 episode 被使用）
    :param pad_before: 每个序列前的填充长度（默认为 0）
    :param pad_after: 每个序列后的填充长度（默认为 0）
    :param debug: 是否启用调试检查（默认为 True）
    
    :return: 返回一个包含索引的数组，每个索引包含四个值：
             [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
             这四个值分别表示从缓冲区中选取的子序列的开始和结束索引，以及在该子序列中的有效部分的起始和结束索引。
    """
    
    # episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # 如果该 episode 不被使用，跳过
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # 循环生成所有可能的子序列索引
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)  # 调试检查
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    """
    根据指定的验证集比例生成验证集掩码。

    :param n_episodes: 训练集中的 episode 数量
    :param val_ratio: 验证集所占比例（0 到 1 之间）
    :param seed: 随机种子（默认为 0）
    
    :return: 返回一个布尔数组，表示哪些 episode 用于验证集（True 表示验证集）。
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # 至少包含 1 个验证集 episode 和 1 个训练集 episode
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    # 随机选择 n_val 个验证集 episode
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    """
    对训练数据进行下采样，保证训练集数量不超过 max_n。

    :param mask: 训练集的布尔掩码，表示哪些 episode 用于训练
    :param max_n: 最大训练集大小
    :param seed: 随机种子（默认为 0）
    
    :return: 返回一个新的训练集掩码，保证其大小不超过 max_n。
    """
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train  # 确保下采样后的训练集大小正确
    return train_mask
