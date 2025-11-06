import pandas as pd

# 读取原始CSV
df = pd.read_csv('neurips_2025_sessions_MexicoCity_detail.csv')

# 匹配结果数据（基于之前的分析）
matching_results = {
    0: {  # Session 1: Socially Responsible and Trustworthy Foundation Models
        'matched_teams': '多伦多云, 计算',
        'relevance_scores': '多伦多云:65, 计算:55',
        'team_research_focus': '多伦多云:AI Agent可靠性与行为管控; 计算:高效模型架构的社会影响',
        'matching_reasons': '多伦多云:Workshop关注trustworthy和accountability机制，直接对应Agent行为可靠性评估与管控需求; 计算:Foundation model的安全设计与模型架构演进趋势相关'
    },
    1: {  # Session 2: Embodied and Safe-Assured Robotic Systems
        'matched_teams': '温哥华云, 计算, 多伦多云, CBG',
        'relevance_scores': '温哥华云:95, 计算:75, 多伦多云:70, CBG:60',
        'team_research_focus': '温哥华云:Embodied AI战略重点领域; 计算:Agentic和多模态前沿应用负载; 多伦多云:具身Agent的可靠性与安全保证; CBG:3D环境感知与重建',
        'matching_reasons': '温哥华云:Workshop完全对应其Embodied/Physical AI战略投资方向; 计算:具身机器人是典型的Agentic多模态前沿负载; 多伦多云:具身系统的安全保证与Agent行为管控高度相关; CBG:机器人需要3D场景理解和重建能力'
    },
    2: {  # Session 3: Research Development AI Mexico
        'matched_teams': '',
        'relevance_scores': '',
        'team_research_focus': '',
        'matching_reasons': '区域性AI应用展示workshop，与所有团队的技术研究方向相关度均低于50分阈值'
    },
    3: {  # Session 4: Vision Language Models: Challenges of Real World Deployment
        'matched_teams': '海思, 计算, 诺亚, 存储, 多伦多云',
        'relevance_scores': '海思:85, 计算:80, 诺亚:75, 存储:70, 多伦多云:60',
        'team_research_focus': '海思:多模态条件融合加速与低延迟推理; 计算:多模态模型架构优化与前沿负载; 诺亚:图文多模态长序列处理; 存储:VLM部署中的模型与KV Cache调度; 多伦多云:Agentic VLM系统集成',
        'matching_reasons': '海思:Workshop聚焦VLM高效推理部署，直接对应多模态融合延迟和实时响应挑战; 计算:VLM是前沿多模态负载，效率优化对计算架构有明确诉求; 诺亚:VLM的长上下文图文推理与多模态长序列技术相关; 存储:VLM部署涉及大模型和KV Cache的加载调度优化; 多伦多云:Workshop明确涵盖Agentic VLM的实际应用'
    },
    4: {  # Session 5: Centering Low-Resource Languages and Cultures
        'matched_teams': '计算, 海思',
        'relevance_scores': '计算:55, 海思:50',
        'team_research_focus': '计算:适应低资源语言的LLM架构设计; 海思:低资源语言LLM的推理加速',
        'matching_reasons': '计算:Workshop涉及针对语言特性的LLM架构定制，与模型架构演进趋势相关; 海思:低资源语言LLM同样需要低延迟高效推理支持'
    },
    5: {  # Session 6: NORA - Knowledge Graphs & Agentic Systems Interplay
        'matched_teams': '多伦多云, 存储, 计算, 诺亚',
        'relevance_scores': '多伦多云:90, 存储:75, 计算:65, 诺亚:55',
        'team_research_focus': '多伦多云:Agent记忆系统与知识图谱集成; 存储:Agent记忆系统(MemOS)的存储访问模式; 计算:KG增强Agentic系统的计算负载; 诺亚:模型内置记忆更新机制',
        'matching_reasons': '多伦多云:Workshop核心议题Agentic系统完全对应其研究方向，KG作为Agent记忆的方案直接相关; 存储:Workshop明确讨论KGs作为agents\' memories，对应MemOS等记忆系统的存储优化; 计算:KG-Agent协同系统是新型Agentic前沿负载; 诺亚:知识图谱的更新维护与模型记忆机制相关'
    },
    6: {  # Session 7: Holistic Video Understanding
        'matched_teams': 'CBG, 诺亚, 计算, 海思, 存储',
        'relevance_scores': 'CBG:85, 诺亚:80, 计算:75, 海思:70, 存储:60',
        'team_research_focus': 'CBG:视频物体分割与长视频生成一致性; 诺亚:视频多模态长序列推理; 计算:视频基础模型架构演进; 海思:Video-LLM长上下文推理加速; 存储:大规模视频训练的IO优化',
        'matching_reasons': 'CBG:Workshop的holistic video understanding与3DAIGC的视频理解需求高度匹配; 诺亚:Video-LLM的长上下文多模态推理是其核心研究方向; 计算:Video foundation models对计算架构提出新挑战; 海思:长视频多模态推理需要高效加速; 存储:大规模视频数据的训练访问模式优化'
    },
    7: {  # Session 8: LLM Persona Modeling
        'matched_teams': '多伦多云',
        'relevance_scores': '多伦多云:55',
        'team_research_focus': '多伦多云:Agent的persona一致性与行为管理',
        'matching_reasons': '多伦多云:Workshop关注LLM persona的一致性和可靠性，与Agent行为可靠性管理有关联'
    },
    8: {  # Session 9: Efficient Transformers (Tutorial)
        'matched_teams': '海思, 计算, 诺亚, 存储',
        'relevance_scores': '海思:90, 计算:85, 诺亚:75, 存储:55',
        'team_research_focus': '海思:稀疏注意力与分块缓存优化; 计算:高效Transformer架构演进; 诺亚:长序列架构的效率优化; 存储:稀疏注意力的内存访问模式',
        'matching_reasons': '海思:Tutorial的sparse attention核心技术完全对应其稀疏注意力与分块缓存研究; 计算:Efficient Transformers直接影响模型架构演进和计算系统设计; 诺亚:Sparse attention和funneling是长序列模型的关键优化方向; 存储:稀疏模式改变内存访问特性'
    },
    9: {  # Session 10: Geospatial Foundation Models (Tutorial)
        'matched_teams': '',
        'relevance_scores': '',
        'team_research_focus': '',
        'matching_reasons': '地理空间基础模型是垂直领域应用，与所有团队的核心研究方向相关度低'
    },
    10: {  # Session 11: Statistically Valid Hyperparameter Selection (Tutorial)
        'matched_teams': '计算',
        'relevance_scores': '计算:50',
        'team_research_focus': '计算:训练范式中的超参数优化',
        'matching_reasons': '计算:超参数选择影响训练效率和模型性能，与训推新范式相关'
    },
    11: {  # Session 12: How to Build Agents to Generate Kernels (Tutorial)
        'matched_teams': '计算, 海思, 多伦多云',
        'relevance_scores': '计算:80, 海思:75, 多伦多云:60',
        'team_research_focus': '计算:GPU kernel优化与计算架构; 海思:LLM推理的kernel加速; 多伦多云:Agent自动代码生成能力',
        'matching_reasons': '计算:Kernel优化是计算架构的核心，直接影响芯片设计和算力效率; 海思:优化的GPU kernel直接提升LLM推理速度; 多伦多云:展示Agent在专业编程任务中的能力'
    },
    12: {  # Session 13: Positional Encoding (Tutorial)
        'matched_teams': '诺亚, 计算, 海思',
        'relevance_scores': '诺亚:75, 计算:70, 海思:60',
        'team_research_focus': '诺亚:长序列位置编码优化; 计算:位置编码对模型架构的影响; 海思:位置编码的计算效率',
        'matching_reasons': '诺亚:Positional encoding是长序列建模的关键，不同编码方式直接影响长序列性能; 计算:位置编码演进(sinusoidal→RoPE等)是模型架构重要变化; 海思:位置编码计算影响推理效率'
    },
    13: {  # Session 14: Science of Trustworthy Generative Foundation Models (Tutorial)
        'matched_teams': '多伦多云',
        'relevance_scores': '多伦多云:60',
        'team_research_focus': '多伦多云:生成式Agent的可信赖性评估',
        'matching_reasons': '多伦多云:Tutorial关注generative models的trustworthiness，与Agent行为可靠性和评估准确性相关'
    },
    14: {  # Session 15: The Oak Architecture (Invited Talk)
        'matched_teams': '温哥华云, 诺亚, 多伦多云, 计算, 存储',
        'relevance_scores': '温哥华云:85, 诺亚:75, 多伦多云:70, 计算:65, 存储:50',
        'team_research_focus': '温哥华云:强化学习架构与RFT; 诺亚:Learning from experience训练框架; 多伦多云:Agent持续学习与自主优化; 计算:RL训推新范式的计算需求; 存储:持续学习的经验数据管理',
        'matching_reasons': '温哥华云:Rich Sutton的model-based RL架构与RFT研究直接相关; 诺亚:Oak的learning from experience核心理念完全对应其研究方向; 多伦多云:持续学习和planning机制是Agent自主优化的关键; 计算:RL范式对计算系统有独特需求; 存储:持续学习需要高效经验管理'
    },
    15: {  # Session 16: Are We Having the Wrong Nightmares About AI? (Invited Talk)
        'matched_teams': '',
        'relevance_scores': '',
        'team_research_focus': '',
        'matching_reasons': '社会学和人文视角的AI影响讨论，与技术研究团队相关度低'
    },
    16: {  # Session 17: The Art of (Artificial) Reasoning (Invited Talk)
        'matched_teams': '诺亚, 温哥华云, 计算, 海思',
        'relevance_scores': '诺亚:85, 温哥华云:75, 计算:70, 海思:55',
        'team_research_focus': '诺亚:后训练Reasoning与RL方法; 温哥华云:RL for LLMs的实践问题; 计算:Reasoning训练范式的计算特征; 海思:推理任务的加速优化',
        'matching_reasons': '诺亚:Yejin Choi讨论的reasoning+RL完全是其核心研究领域; 温哥华云:Talk分析RL在reasoning中的成功与挑战，对RFT有直接指导; 计算:Reasoning范式影响训练负载特征; 海思:推理任务的加速优化对推理加速有意义'
    },
    17: {  # Session 18: Evaluating Cognitive Capabilities (Invited Talk)
        'matched_teams': '多伦多云',
        'relevance_scores': '多伦多云:65',
        'team_research_focus': '多伦多云:Agent认知能力评估方法学',
        'matching_reasons': '多伦多云:Talk提出的认知评估方法论可应用于Agent评估，对应其\'提升Agent评估准确性\'的研究需求'
    },
    18: {  # Session 19: From Benchmarks to Problems (Invited Talk)
        'matched_teams': '',
        'relevance_scores': '',
        'team_research_focus': '',
        'matching_reasons': '研究方法论和问题选择的元讨论，与具体技术研究方向相关度低'
    },
    19: {  # Session 20: Demystifying depth (Invited Talk)
        'matched_teams': '计算',
        'relevance_scores': '计算:65',
        'team_research_focus': '计算:深度网络学习动力学与架构演进',
        'matching_reasons': '计算:深度学习理论有助于理解模型架构演进的根本驱动力，支持计算架构的前瞻性设计'
    }
}

# 添加新列
df['matched_teams'] = ''
df['relevance_scores'] = ''
df['team_research_focus'] = ''
df['matching_reasons'] = ''

# 填充匹配结果
for idx, result in matching_results.items():
    df.at[idx, 'matched_teams'] = result['matched_teams']
    df.at[idx, 'relevance_scores'] = result['relevance_scores']
    df.at[idx, 'team_research_focus'] = result['team_research_focus']
    df.at[idx, 'matching_reasons'] = result['matching_reasons']

# 保存增强后的CSV
output_file = 'neurips_2025_sessions_MexicoCity_detail_matched.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"匹配完成！已生成文件: {output_file}")
print(f"\n统计信息:")
print(f"- 总会议数: {len(df)}")
print(f"- 有匹配团队的会议数: {(df['matched_teams'] != '').sum()}")
print(f"- 无匹配团队的会议数: {(df['matched_teams'] == '').sum()}")

# 统计每个团队的匹配次数
team_counts = {}
for teams_str in df['matched_teams']:
    if teams_str:
        for team in teams_str.split(', '):
            team_counts[team] = team_counts.get(team, 0) + 1

print(f"\n各团队匹配统计:")
for team, count in sorted(team_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {team}: {count}个sessions")
