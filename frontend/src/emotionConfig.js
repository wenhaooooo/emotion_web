// emotion2vec 9-class labels (default)
export const E2V_COLORS = {
  0: '#FF3B30',  // angry
  1: '#A2845E',  // disgusted
  2: '#AF52DE',  // fearful
  3: '#FF9500',  // happy
  4: '#007AFF',  // neutral
  5: '#8E8E93',  // other
  6: '#5856D6',  // sad
  7: '#FFCC00',  // surprised
  8: '#C7C7CC',  // unknown
}

export const E2V_NAMES = {
  0: 'angry',
  1: 'disgusted',
  2: 'fearful',
  3: 'happy',
  4: 'neutral',
  5: 'other',
  6: 'sad',
  7: 'surprised',
  8: 'unknown',
}

export const E2V_NAMES_CN = {
  0: '愤怒',
  1: '厌恶',
  2: '恐惧',
  3: '开心',
  4: '中性',
  5: '其他',
  6: '悲伤',
  7: '惊讶',
  8: '未知',
}

// Teacher 4-class labels (optional)
export const TEACHER_COLORS = {
  0: '#FF9500',  // enthusiastic
  1: '#007AFF',  // calm
  2: '#8E8E93',  // negative
  3: '#FF3B30',  // tense
}

export const TEACHER_NAMES = {
  0: '热情投入',
  1: '平稳中性',
  2: '消极低落',
  3: '紧张焦虑',
}

export function getColors(mode) {
  return mode === 'teacher' ? TEACHER_COLORS : E2V_COLORS
}

export function getNames(mode) {
  return mode === 'teacher' ? TEACHER_NAMES : E2V_NAMES_CN
}

export function getLabel(seg, mode) {
  return mode === 'teacher' ? seg.label : seg.emotion2vec_label
}

export function getLabelName(seg, mode) {
  return mode === 'teacher' ? seg.label_name_cn : (E2V_NAMES_CN[seg.emotion2vec_label] || seg.emotion2vec_label_name)
}
