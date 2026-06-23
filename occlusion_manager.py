# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

@dataclass
class BehaviorTimer:
    """行为计时器，支持累计、挂起与恢复"""
    start_time: float = 0.0             # 本次计时开始时间戳
    accumulated_time: float = 0.0       # 历史已累计的有效时长（秒）
    is_running: bool = False            # 是否处于计时状态
    is_suspended: bool = False          # 是否处于遮挡挂起状态
    suspend_time: float = 0.0           # 挂起发生的时间戳


@dataclass
class StaticSpatialAnchor:
    """静态空间锚点 - 用于强遮挡下的计时继承"""
    anchor_id: int
    world_pos: Tuple[float, float]      # 物理世界的 2D 坐标 (X_real, Y_real)
    radius: float = 0.8                 # 锚点有效匹配半径（米）
    original_track_id: int = -1         # 创建该锚点的原始 track_id
    identity: str = None                # 绑定的衣服/安全帽号码
    accumulated_time: float = 0.0       # 挂起前累计的静止时间
    start_time: float = 0.0             # 原始静止行为的开始时间
    last_associated_time: float = 0.0   # 锚点最后一次有人停留的时间戳
    ttl: float = 15.0                   # 锚点丢失后的存活期限（秒）
    is_active: bool = True              # 锚点是否有效


class TrackedPerson:
    """每个被追踪人员的业务状态"""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.identity = None            # 衣服号码数字
        self.bbox = None                # 当前像素边界框 [xmin, ymin, xmax, ymax]
        self.world_pos = None           # 物理世界 2D 坐标
        self.velocity = 0.0             # 物理空间中的瞬时速度 (米/秒)
        self.is_occluded = False        # 是否正处于遮挡状态
        self.last_seen_time = 0.0       # 最后一次看到该人员的时间戳
        
        # 行为计时器
        self.still_timer = BehaviorTimer()
        self.headdown_timer = BehaviorTimer()


class OcclusionManager:
    """
    遮挡与计时继承管理器 (核心业务降噪算法)。
    实现空间静态锚点的创建、匹配、计时继承和超时清理。
    """
    def __init__(self):
        self.anchors: Dict[int, StaticSpatialAnchor] = {}
        self.anchor_counter = 0

    def update_anchors(self, tracked_persons: Dict[int, TrackedPerson], current_time: float):
        """
        每帧处理主循环：
        1. 针对新出现的、速度极低的 track_id，检查是否可继承邻近的空间锚点。
        2. 针对已经处于静止状态且达到创建阈值的人员，为其维护/创建物理空间静态锚点。
        3. 对断开/丢失的目标，如果超时未继承则销毁锚点。
        """
        # --- Step 1: 处理新 ID 的计时继承 ---
        for track_id, person in tracked_persons.items():
            if person.world_pos is None:
                continue
                
            # 继承触发条件：人员处于静止状态（速度极低） 且 静止计时器尚未运行 且 过去1秒内刚被创建/重新激活
            is_new_still = (person.velocity < 0.15) and (not person.still_timer.is_running)
            
            if is_new_still:
                # 寻找物理距离最近且在 TTL 窗口内的空闲锚点
                best_anchor = None
                min_dist = float('inf')
                
                for anchor in self.anchors.values():
                    if not anchor.is_active:
                        continue
                    # 检查时间容差
                    time_diff = current_time - anchor.last_associated_time
                    if time_diff <= anchor.ttl:
                        # 计算欧氏物理距离（米）
                        dist = np.linalg.norm(np.array(person.world_pos) - np.array(anchor.world_pos))
                        if dist < anchor.radius and dist < min_dist:
                            min_dist = dist
                            best_anchor = anchor
                
                # 匹配成功，执行继承
                if best_anchor is not None:
                    # 继承静止计时器
                    person.still_timer.accumulated_time = best_anchor.accumulated_time
                    person.still_timer.start_time = best_anchor.start_time
                    person.still_timer.is_running = True
                    person.still_timer.is_suspended = False
                    
                    # 继承身份
                    if best_anchor.identity:
                        person.identity = best_anchor.identity
                        
                    # 更新锚点关联状态为当前新 ID
                    best_anchor.original_track_id = track_id
                    best_anchor.last_associated_time = current_time
                    print(f"[OcclusionManager] 追踪 ID {track_id} 成功继承静态锚点 {best_anchor.anchor_id} 的历史静止时间 ({best_anchor.accumulated_time:.1f} 秒)，身份重绑定为: {person.identity}")
                    
                # 如果没有匹配的锚点，且此人确实已经原地站了一小会儿，则为其启动新的计时
                elif not person.still_timer.is_running:
                    person.still_timer.start_time = current_time
                    person.still_timer.accumulated_time = 0.0
                    person.still_timer.is_running = True
                    print(f"[OcclusionManager] 追踪 ID {track_id} 在 {person.world_pos} 处开始计时静止")

        # --- Step 2: 维护/创建静态空间锚点 ---
        for track_id, person in tracked_persons.items():
            if person.world_pos is None or not person.still_timer.is_running:
                continue
                
            # 计算总静止时间
            elapsed = current_time - person.still_timer.start_time + person.still_timer.accumulated_time
            
            # 条件：已经静止超过 5 秒，则为其在物理位置上锚定一个 StaticSpatialAnchor
            if elapsed > 5.0:
                # 寻找该 ID 已经拥有的锚点
                associated_anchor = None
                for anchor in self.anchors.values():
                    if anchor.original_track_id == track_id and anchor.is_active:
                        associated_anchor = anchor
                        break
                
                if associated_anchor is not None:
                    # 更新已有锚点的信息
                    associated_anchor.world_pos = person.world_pos
                    associated_anchor.accumulated_time = elapsed
                    associated_anchor.last_associated_time = current_time
                    if person.identity:
                        associated_anchor.identity = person.identity
                else:
                    # 创建全新锚点
                    self.anchor_counter += 1
                    new_anchor = StaticSpatialAnchor(
                        anchor_id=self.anchor_counter,
                        world_pos=person.world_pos,
                        original_track_id=track_id,
                        identity=person.identity,
                        accumulated_time=elapsed,
                        start_time=person.still_timer.start_time,
                        last_associated_time=current_time
                    )
                    self.anchors[new_anchor.anchor_id] = new_anchor
                    print(f"[OcclusionManager] 为 ID {track_id} 新建物理空间静态锚点 {new_anchor.anchor_id}，物理坐标: {person.world_pos}")

        # --- Step 3: 清理过期未继承的锚点 ---
        expired_ids = []
        for aid, anchor in self.anchors.items():
            if not anchor.is_active:
                continue
            
            # 检查关联的 track_id 是否已经在场
            orig_id = anchor.original_track_id
            is_lost = True
            
            if orig_id in tracked_persons:
                person = tracked_persons[orig_id]
                # 如果该人还在场，且未被标记为丢失/遮挡，则不算丢失
                if not person.is_occluded and (current_time - person.last_seen_time < 2.0):
                    is_lost = False
            
            if is_lost:
                # 触发 TTL 超时倒计时
                time_lost = current_time - anchor.last_associated_time
                if time_lost > anchor.ttl:
                    # 再次确认物理位置上是否有其他行人在
                    anyone_present = False
                    for person in tracked_persons.values():
                        if person.world_pos is not None:
                            dist = np.linalg.norm(np.array(person.world_pos) - np.array(anchor.world_pos))
                            if dist < anchor.radius and person.velocity < 0.2:
                                anyone_present = True
                                break
                    # 如果该区域确实没人占据了，彻底注销此锚点
                    if not anyone_present:
                        anchor.is_active = False
                        expired_ids.append(aid)
                        print(f"[OcclusionManager] 静态锚点 {aid} 超时未被继承，且原地无静止人员，正式销毁。")

        for aid in expired_ids:
            del self.anchors[aid]
