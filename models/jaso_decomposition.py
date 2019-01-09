# coding: utf8
import re

LEADING = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
VOWEL = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
TRAILING = 'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'
SEP_TRAILING = ["ㄱ", "ㄲ", "ㄱㅅ", "ㄴ", "ㄴㅈ", "ㄴㅎ", "ㄷ", "ㄹ", "ㄹㄱ", "ㄹㅁ", "ㄹㅂ", "ㄹㅅ", "ㄹㅌ", "ㄹㅍ", "ㄹㅎ", "ㅁ", "ㅂ", "ㅂㅅ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
TCount = len(TRAILING) + 1
VTCount = len(VOWEL) * TCount

def decompose(s_text, s_null_coda_replacer = '', b_remove_non_hangeul=True,
 b_remove_onset_ieung=False, s_ieung_replacer = "", b_sep_trailing = True, b_return_tuple=False, r_compiled_filter=False):
	'''한글 음절을 유니코드 한글 호환성 자모 구역의 문자로 분해하는 함수

	>>> decompose('밥')
	'ㅂㅏㅂ'
	>>> decompose('반찬')
	'ㅂㅏㄴㅊㅏㄴ'

	# 받침 자리 비우기/채우기
	>>> decompose('두유')
	'ㄷㅜㅇㅠ'
	>>> decompose('두유', '_')
	'ㄷㅜ_ㅇㅠ_'

	# 한글 음절 이외의 문자 제거하기/유지하기
	>>> decompose('1박2일', b_remove_non_hangeul = False)
	'1ㅂㅏㄱ2ㅇㅣㄹ'
	>>> decompose('1박2일')
	'ㅂㅏㄱㅇㅣㄹ'

	# 문자열이 아니면 반환이 없음
	>>> decompose(42)
	'''
	# 인수의 자료형이 문자열인 경우에만 반환을 시도하기
	if type(s_text) == str:
		try:
			if ord('가') <= ord(s_text) <= ord('힣'):
				# 한글 한 음절인 경우
				ind = ord(s_text) - ord('가')
				L = LEADING[ind // VTCount] # 초성
				if b_remove_onset_ieung and L == "ㅇ":
					L = s_ieung_replacer
				V = VOWEL[ind % VTCount // TCount] # 중성
				if b_sep_trailing:
					T = SEP_TRAILING[ind % TCount - 1] if ind % TCount else s_null_coda_replacer
				else:
					T = TRAILING[ind % TCount - 1] if ind % TCount else s_null_coda_replacer
				if b_return_tuple:
					return (L,V,T)
				else:
					return ''.join((L,V,T))
			else:
				# 한글 음절이 아닌 문자열인 경우
				return '' if b_remove_non_hangeul else s_text
		except:
			# 길이 2 이상의 문자열인 경우 ord()에서 TypeError 발생
			if r_compiled_filter:
				s_text = r_compiled_filter.sub("", s_text)
			return ''.join(
				decompose(char, s_null_coda_replacer, b_remove_non_hangeul, b_remove_onset_ieung, s_ieung_replacer, b_sep_trailing)
					for char in s_text)
	else:
		return


def find_choseong(s_text):
	try:
		if ord('가') <= ord(s_text) <= ord('힣'):
			# 한글 한 음절인 경우
			ind = ord(s_text) - ord('가')
			L = LEADING[ind // VTCount] # 초성
			return L
	except:
		pass # 한글 한 음절이 아닌 경우
			
def find_jongseong(s_text):
	try:
		if ord('가') <= ord(s_text) <= ord('힣'):
			# 한글 한 음절인 경우
			ind = ord(s_text) - ord('가')
			if ind % TCount == 0:
				T = '' # 종성이 없을 경우
			else:
				T = TRAILING[ind % TCount - 1] # 종성
			return T
	except:
		pass # 한글 한 음절이 아닌 경우

# if __name__ == '__main__':
# 	tongue1 = '아무렴 닭알이 저기 있네'
# 	print(decompose(tongue1, b_remove_onset_ieung=True, b_sep_trailing=True))