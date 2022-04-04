from lib2to3.pgen2 import token
import tensorflow as tf
from transformers import BertTokenizerFast
from dataloader.tfrecord_utils import  serialize_ids, masked_lm_predictions
import os
from tqdm import tqdm
import math
import random

file_from = 'data/sample.txt'
file_to = 'data/sample.tfrecord'

tokenizer = BertTokenizerFast.from_pretrained('./tknzrs/daily_tknzr')


def tokenize_line(line, max_len=256, csep_token_id=5):
    input_ids = tokenizer.encode(line)
    if len(input_ids) < max_len:
        return [input_ids]
    holder = []
    bos, eos = input_ids[0], input_ids[-1]
    def cut_and_append(line):
        if len(line) < max_len:
            holder.append(line)
            return
        csep_loc = [index for index, element in enumerate(line) if element == csep_token_id]
        if len(csep_loc) == 0:
            picked_idx = random.randint(len(line) // 3, len(line) * 2 //3)
            input_a, input_b = line[:math.floor(picked_idx + 0.1*max_len)], line[math.ceil(picked_idx - 0.1*max_len ):]
        else:
            pick = random.randint(len(csep_loc)//3, len(csep_loc) * 2 //3)
            picked_idx = csep_loc[pick]
            input_a, input_b = line[:picked_idx], line[picked_idx + 1:]
        cut_and_append(input_a + [eos])
        cut_and_append([bos] + input_b)

    cut_and_append(input_ids)
    return holder

def replace_sep(sent):
    return sent.replace('#|#', '[MSEP]').replace('#&#', '[CSEP]').replace('#S#', '[SEP]')

def write_tfrecord(files_from, out_dir, mode='gpt', seed=None, **kwargs):
    def get_file_path(file_idx):
        if seed is None:
            fname = f'record_{file_idx}.tfrecord'
        else:
            fname = f'seed_{seed}_record_{file_idx}.tfrecord'
        return os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
    char_cnt_max = 100_000_000
    char_cnt = 0 
    file_idx = 1
    writer = tf.io.TFRecordWriter(get_file_path(file_idx))
    for fpath in files_from:
        with open(fpath, 'r') as fr:
            for i, line in enumerate(tqdm(fr)):
                if char_cnt > char_cnt_max:
                    print(i // 10000, '만')
                    writer.close()
                    file_idx += 1
                    char_cnt = 0
                    writer = tf.io.TFRecordWriter(get_file_path(file_idx))
                char_cnt += len(line)
                line = replace_sep(line)
                if mode == 'gpt':
                    examples = process_gpt(line)
                elif mode == 'mlm':
                    examples = process_mlm(line, kwargs['with_sop'])
                elif mode == 'ctx':
                    examples = process_ctx(line)
                else:
                    raise Exception(f'mode {mode} not supported!')

                if not examples:
                    print(examples)
                    continue
                for example in examples:
                    writer.write(example)
    writer.close()

def process_ctx(line):
    try:
        context, ans = line.split('[SEP]')
        context_ids = tokenize_line(context)[-1]
        ans_ids = tokenize_line(ans)[-1]
    except:
        print(line)
        return None
    example = serialize_ids(context_ids=context_ids, answer_ids=ans_ids)
    return [example]

def process_mlm(line, with_sop):
    holder = []
    input_ids_list = tokenize_line(line)
    for input_ids in input_ids_list:
        mout = masked_lm_predictions(input_ids, 0.2, with_sop=with_sop)
        if with_sop:
            example = serialize_ids(masked_input_ids = mout['masked_input_ids'], masked_label=mout['masked_label'], sop_label=mout['sop_label'])
        else:
            example = serialize_ids(masked_input_ids = mout['masked_input_ids'], masked_label=mout['masked_label'])
        holder.append(example)
    return holder

def process_gpt(line):
    holder = []
    input_list = tokenize_line(line)
    for input_ids in input_list:
        example = serialize_ids(input_ids=input_ids)
        holder.append(example)
    return holder



if __name__ == "__main__":
    # file_name = 'news2'
    # with_sop = True
    # for i in range(0, 3):
    #     write_mlm_tfrecord([f'data/bert/mlm_data/{file_name}.txt'], f'data/bert/{file_name}/', seed=i, with_sop=with_sop)
    # write_mlm_tfrecord(['data/bert/mlm_data/news.txt'], 'data/bert/news', seed=4)
    # write_mlm_tfrecord(['data/sample.txt'], 'data/sample', with_sop=True)

    write_tfrecord(['data/gpt/gpt_data/everytime.txt'], 'data/gpt/everytime/', mode='gpt')
    # write_ctx_tfrecord(['data/transformer/context_data/everytime.txt'], 'data/transformer/everytime/')


''' test tokenize'''
    # sent = '오늘 농구할 솨람~~#&#나 10시 이후면 생각 있는데#|#넘 늦나?#&#엇#|#넘 늦을듯ㅜㅜ#&#그럼#|#만약 홍제천에서 하면#|#퇴근하고 바로 글로 감#|#만약 하면ㅋㅋ#&#혹시#|#내일이나 모레는#|#어떠하냐#|#나는 송도기때문이지#&#2월 6일은#|#우쨔됨#|#재현친구가 와야 할 수 있는데ㅔ#&#@김재현 에비슨ra#|#현우 훈련소가기전에#|#많이 해놔야하는데#|#빡빡이되면 겸상하기 부끄럽잖어ㅠㅠ#&#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ#|#모자쓰고 오겠지#&#이자식들#&#ㅋㅋㅋㄱㅋㅋㅋ#&#항상 데꼬 다녀라#&#ㅈㅅ#&#빡빡이~~#&#오 저도 2월 6일 좋아요!#&#오예#&#구뜨#&#농구할사람#|#나#&#ㅌㅋㅋㅋㅋ#|#ㅇㅈ#&#오늘 운동 안했으면 갔는데#|#ㅈㅅ#|#언제 머리 미냐#&#ㄱ?#|#ㅋㅋㅋㅋ#|#나 집치우고 있긴한대#|#난 가능#|#운동했으면#|#영준씨#|#그 유산소하실 차례아닙니꺼?#&#좀 거리감이 느껴지네요#|#현탁씨#|#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ#|#그#|#유산소는#&#컨셉을 이제 버릴 수 없습니다#&#바이크 탔어요#|#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ#&#오 하프타임에 바이크#|#타던데#|#농구선수들은#&#ㅋㅋ;;#&#역시 예열!#&#이걸 웜업으로..?#|#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ#&#바이크 탔으면 농구 해줘야지#&#언제 할건대#|#한명 더 없냐#|#2:2하게#&#오늘 군희 꼬셔오면 내가 치킨 사줌#&#ㄹㅇ#&#선착순#&#나 아직 밥안먹은거 어케 알고?#&#나도#&#군희야 가쟈#&#ㅎ#&#아직 안먹음#|#군희#|#바쁘던디#|#재현친구는#|#뭐하는데#&#운동도하면서#|#일해야지 군히씨#&#재현이는 ㄹㅇ 바쁨ㅠㅠ#|#로스쿨생#&#제현이는 그저께 제 전여친이랑 봤다던데 ㅠㅠ#&#빨리#&#형은 버렸냐?#|#ㅠㅠㅠ#&#한명 더 구해봐#|#전여친….?#|#아니 소개팅 잘 안됨?#&#아니 제 전여친이요#|#ㅋㅋㅋㅋㅋㅋㅋㅋㅋ#&#?#&#재현이 친구임#&#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅌㅋ#&#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ#|#아니#|#나 진심 궁금한데#&#나쁘다 김제현!#&#출현하셨네#|#아니 제발#|#우째됨???#&#재밌었어용!#&#아 ㅇㅋ#|#접수 완료#|#재미만 있었던걸로#&#이모티콘#&#쿨가이#|#ㅋㅋㅋ#&#까빙;#|#아니 근데 진짜#|#근처에 농구할만한 친구 없음??#|#1:1:1은#|#지금 날씨에 좀 힘든#|#쉬는 사람 바로 손 얼어#&#날씨 좋던디#&#날씨 오늘 따뜻하던데#|#가자가자갖갖갖갖갖가자#|#언제 가실?#|#8시 출발 ㄱ?#&#아직 퇴근 안했는디#|#8시반 ㄱㄱ#&#누구 없나#|#어이#|#연대생들#&#없음#&#사람좀 구해 봐#&#가면 잘하는사람 개많음#|#걱정 ㄴㄴ#&#아 ㅇㅋ#|#발릴준비 하면 됨?#&#가면 장난아니고 우리 하는 거 2배로 뛴다고 보면 됨#|#근데 그렇게 하기 싫은데#|#ㅠㅠ#&#미투#|#힘듬#&#일단 ㄲ#&#오늘#|#각 아닌들#|#좀 뺌#|#ㅈㅅ#&#ㅇㅋㅇㅋ#&#담에 같이 ㄲ#|#내일 ?#&#내일 될듯#&#좀 빨리 가능?#|#7시쯤#&#ㄱㄴ#&#근데 다들 바쁘나#|#ㅇㅂㅇ#&#삭제된 메시지입니다.#|#나는 가능ㅋㅋ#&#ㅋㅋㅋㅋ#|#아니면 오늘 열시?#&#헐#&#기?#&#ㄱㄱ?#&#긔긔?#&#ㄱㄱ#&#ㄱㄱ#&#ㄲ#&#난 ㄱㅊ#&#영준이#|#가능#&#@위영준#&#하지?#&#ㄱㄴ#|#@위영준#&#???#|#나 씻고 나왔는데#&#ㄱㄱ!#&#다시 먼지 묻히러 가자#&#??#|#봐줘;#|#잘준비 끝냈는데;#&#버핏하우스 앖에서 보까#|#현우야#|#나지금 나옴ㅋㅋ#&#먼저 가게?#|#나 10시에 출발할게ㅋㅋㅋ#&#ㅇㅇㅋㅋㅋ#&#…..?#|#왜 버핏하우스??#&#영준아#&#?#&#전화 끊고 더자#|#가자#&#여자친구 전화임#&#너네 집앞이야#|#나와#&#??#&#나오셈#|#안갈겨?#&#10시 아님?#&#10시에 올거임?#|#나 그럼 먼저 가있게ㅋㅋ#&#ㅇㅇ#&#ㅇㅋ#&#얘들아#|#오지마#&#??#&#현우 부상#&#헐#|#미친#&#??#&#많이 다침?#&#뭔일이여#&#무릎 다침#&#어이고#&#…#&#....#|#많이 다쳣나>#&#심각해?#&#ㅇㅇ#&#병원 가야하지 않나?>#&#ㄴㄴ#&#미친#|#응급실가;#&#안심각#&#그럼 가고#&#현탁이 똥싸러감#&#?#|#뭐임#&#근데 뛰진 못하고#&#누구 말을 믿어야 하는거냐#&#터렛은 할 수 잇을듯#&#그럼#|#심각한거 아니냐#&#키 170 터렛 가능?#&#못뛰는거면#|#쌉불가능#&#ㅜㅜ#&#그#|#병원은#|#부활용도로 가는게 아니라#|#다치면 가는곳이란다#|#빨리 병원을 가렴#&#다친건 아닌데#|#뛰면 아파#&#그게#|#사회적으로 부상이라고 합의함#&#그냥 와서 노셈#&#ㅇㅇ#|#아니#&#사람들이랑 하면 할만 함#&#근데#|#진짜#|#ㄱㅊ?#&#사람들 ㅈㄴ 잘해#&#ㅇㅇ#|#슛 쏠 수 있음#&#천천히 걸어다녀#&#난 슛쏘러 간다#&#곧 출발#|#ㅇㅇ#&#방금 15대 5로 발렸나?#&#너네 오고시ㅠ음 오고#&#ㅋㅋㅋㅋㅋㅋㅋ#&#아니면 집에서 쉬삼#&#15대5는 뭔데#&#점수#&#ㅋㅋㅋㅋㅋ#|#우쩌지#|#영준이 가는겨?#&#근데 오면#|#같이할사람은 많어#&#흠#&#일단 난 응아중#&#고민되는군#|#나는 사실#|#심하게 하고싶진 않아서#&#현우 빨리 드가봐야되는거아니냐#&#우리끼리 2:2하려고 했지#&#글게#|#아숩네..#&#그냥 산책식으로#|#와도 되긴하는데#|#사실 안와도 됨#|#ㅋㅋㅋ#&#ㅇㅇ#&#ㅋㅋㅋㅋㅋㅋㅋㅋㅋ#&#군희#&#편하게 ㄱㄱ#&#우짤겨#&#안갈게#|#ㅎ#&#그럼#|#나도 오늘은 쉴께#|#ㅈㅅㅈㅅ#|#내일 7시에#|#간다#&#ㅇㅋ#|#현우 얼른 쉬어라#&#그니까#|#그 병원이란 곳을#&#냉찜질이라도 하고#&#현우#&#가보는것도 나쁘지 않을것같아#&#슛던지로간듯#&#미친놈이군#&#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'
    # sent = '''
    # “갑오년 새해에는 한 살 어려보이게 해주세요” SBS 드라마 '상속자들' 속 김성령에서 tvN '꽃보다 누나'의 김희애, 이미연으로 이어진 중년 여배우 열풍으로 인해 2014년에도 '동안(童顔)' 트렌드는 계속될 전망이다. 동안이란 나이에 비해 어려보이는 외모를 일컫는 말. 이러한 트렌드를 반영하듯 최근에는 '내 피부 나이 테스트'라는 테스트를 통해 자신이 동안인지 노안인지를 스스로 평가하는 것이 유행하고 있다. 큰 눈, 도톰한 입술 등 동안으로 거듭나기 위한 많은 조건 중 무엇보다 중요한 것은 단연 주름없이 깨끗하고 탄력 넘치는 피부다. 동안의 핵심 요소 '피부'에 초점을 맞춘 동안피부 만드는 방법을 소개한다. Part 1 내 피부 나이 테스트 “나는 동안 피부일까 노안 피부일까?” 결과의 수치를 나이에 합산한 것이 본인의 실제 피부 나이다. 결과가 +5일 경우 본래 나이보다 5살 많은 피부를 가지고 있다는 것이며 -5인 경우 5살 어린 동안 피부를 지니고 있다는 의미. 만약 이 결과치가 +10이 넘어갈 경우 노화가 급격히 진행될 가능성이 매우 높으므로 각별한 주의가 필요하다. Part 2 동안이 되기 위한 피부 관리법 STEP1 동안 세안법 “동안으로 가는 첫 걸음은 세안이다” 올바른 동안피부 세안법은 어떠한 방식으로도 피부에 자극을 주지 않는 것이다. 손의 세균이 얼굴로 전이되지 않도록 올바른 동안 세안을 위해서는 세안 전 비누로 손을 깨끗이 씻어주는 단계가 우선되어야 한다. 그 다음 클렌징폼을 이용해 풍성한 거품으로 피부에 자극을 주지 않도록 세안을 해주자. 마치 손과 피부 사이에 미세한 공간이 있다는 느낌으로 세안하면 손바닥으로 인한 마찰 자극을 줄여 더욱 부드러운 세안이 가능하다. 이후 미온수를 이용해 거품 잔여물을 깨끗이 씻어낸 뒤 차가운 물로 30회 이상 헹구어줄 것. 차가운 물은 피부를 자극하고 모공을 수축시켜 더욱 탄력 있는 피부로 거듭날 수 있도록 돕기 때문에 동안피부를 위해서는 결코 잊어서는 안 될 단계다. STEP2 기초화장품으로 피부를 건강하게 케어하자 세안을 마친 직후 피부에 남아있는 수분이 모두 증발하는데 걸리는 시간은 채 5분이 되지 않는다고 한다. 이를 그대로 방치할 경우 피부 건조함을 유발할 수 있으므로 미스트나 세럼 등을 이용해 세안 후 피부에 즉각적인 영양분을 공급해주자. 이후 동안피부 관리에 효과적인 성분이 함유된 크림 제품을 이용해 피부 전체에 영양분을 공급해줄 것. 동안 피부의 조건인 탄력과 볼륨감을 케어할 수 있는 성분으로는 불가사리가 가장 대표적이다. 불가사리는 몸을 여러 조각으로 잘라도 죽지도 살아난다 하여 불가사리라 불리는 생명체. 이 같은 불가사리의 재생력을 이용해 꾸준히 관리해주면 메이크업 후는 물론 민낯도 완벽한 동안피부로 거듭날 수 있을 것이다. 불가사리 크림을 고를 때는 불가사리 성분이 얼마나 함유되어 있는지를 꼼꼼하게 체크해야 한다. 불가사리 성분이 70%이상 함유된 제품을 골라야 불가사리의 재생력이 피부 깊숙이 전달될 수 있으므로 참고하자. STEP3 일주일에 세 번 스타존을 케어하라 '스타존'은 동안이냐 노안이냐를 결정하는 얼굴의 중심을 일컫는다. 눈가, 미간, 입가 주름까지 얼굴의 중심을 잡아주는 표정주름이 탱탱하게 펴져 있기만 해도 평균 나이보다 다섯 살은 어려보일 수 있기 때문. 스타존을 보다 확실하게 케어하기 위해 아이크림과 더불어 '불가사리 마스크 팩'이라 불리는 별 모양의 마스크팩을 활용해보자. 스타존의 표정 주름은 물론 칙칙한 눈가를 캐어해 환하면서도 탄력 있는 피부로 거듭날 수 있도록 도움을 받을 수 있을 것이다. 그러나 불가사리 마스크 팩을 지나치게 장시간 부착하고 있는 것은 금물. 권장 사용 시간을 넘을 경우 오히려 피부가 건조해질 수 있으므로 15분 내지 20분 후 떼어내고 잔여물을 가볍게 두드려 흡수시켜주는 것으로 마무리하자. Part 3 동안 피부로 거듭나기 위한 마사지 동안 피부로 거듭나기 위해서는 꾸준한 기초제품 사용을 통한 케어와 더불어 마사지가 병행되어야 한다. 그러나 손이 더러우면 세균으로 인해 피부에 악영향을 끼칠 수 있으므로 손을 깨끗하게 씻은 상태에서 마사지할 것. 01 양손을 주먹 쥔 뒤 코의 양옆 뺨 위에 가볍게 올려놓는다. 02 얼굴에 힘을 뺀 상태에서 볼, 귀 옆까지 힘을 주어 쓸어 올린다. 03 위의 동작을 10회 이상 반복한 뒤 턱 살을 가볍게 꼬집어주며 마무리한다. Part 4 동안 피부를 위한 '머스트 해브 뷰티 아이템' 01 뉴트로지나 딥클린 포밍 클렌저 크림 같은 느낌의 독특한 거품으로 모공 속 과다 피지와 노폐물을 분해하고 씻어주는 클렌징폼으로 롱래스팅 오일 컨트롤 성분이 피부의 유분을 케어해 부드럽고 매끈한 피부를 유지해준다. 02 메이크업포에버 미스트 앤 픽스 수분을 공급해주고 메이크업을 고정시켜주는 광택 미스트로 메이크업 후 픽서로 사용할 수 있는 것은 물론 세안 직후 피부에 즉각적인 수분을 공급하는 미스트로 사용할 수 있다. 03 에스티로더 나이트리페어 세럼 일명 '갈색병 리페어'라 불리는 세럼으로 히알루론산과 에스티로더만의 크로노룩스 기술이 피부 리듬을 바로 잡아 피부가 스스로 개선하는 능력을 되살려줌으로써 동안 피부로 거듭날 수 있도록 돕는다. 04 미즈온 리터닝 스타피쉬 크림 불가사리 추출물을 70% 함유한 안티에이징 크림으로 특유의 형상기억포뮬라로 인해 '밀당크림'이라는 애칭을 갖고 있기도 하다. 고기능성 크림으로 피부에 수분과 탄력, 주름 개선 등 복합적인 피부 문제를 해결하며 동안 피부 관리에 효과적이다. 05 미즈온 리터닝 스타피쉬 바이오 마스크 스타존 케어를 위해 특수 제작된 불가사리 마스크로 눈가, 입가, 미간에 리프팅 효과를 부여한다. 일명 '미즈온 불가사리 마스크 팩'이라 불리는 제품으로 불가사리 추출물이 60% 함유되어 확실한 동안 케어를 돕는다.
    # '''
    # sent = replace_sep(sent)
    # holder = tokenize_line(sent, max_len=100)

    # for sent in tokenizer.batch_decode(holder):
    #     print('-----')
    #     print(sent)