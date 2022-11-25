import React, {useEffect} from 'react';
import { Controller, Scene } from 'react-scrollmagic';
import {Button} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import Aos from 'aos';

import 'css/everytime.css';
import 'aos/dist/aos.css';


export default function EverytimeScreen() {
  const navigate = useNavigate();

  useEffect(() => {
    Aos.init({duration: 1000});
    Aos.refresh();
  }, []);

  return (
    <div id='everytime-screen'>
      {/* page 1 */}
      <div className='section s1'>
        <div data-aos='fade-up'
          className='animation-box'
          style={{
            top: '20%',
            width: '60%',
            left: '52%',
            transform: 'translate(-50%, 0)',
          }}
        >
          <span>대학생들이 이용하는 익명 커뮤니티</span>
        </div>
        <div className='max-width'>
          <div className='img-row'>
            <img className='et-img' src='images/everytime1.png'/>
            <img className='et-img' src='images/everytime2.png'/>
          </div>
        </div>
      </div>


      <Controller globalSceneOptions={{triggerHook: 'onLeave'}}>
        <Scene pin>
          <div className="section s2">
            <div data-aos='fade-right'
              className='animation-box'
              style={{
                top: '30%',
                width: '60%',
                left: '52%',
                transform: 'translate(-50%, 0)',
              }}
            >
              <h1>AI를 커뮤니티 게시글로 학습시키면 무슨 말을 할까?</h1>
            </div>
          </div>
        </Scene>
        <Scene pin>
          <div className="section s3">
            <GeneratedResult
              query='남친 사귀는 팁'
              results={[
                '남친 사귀는 팁좀. 내가 먼저 고백했는데 ( 여자 ) 다른 남자들이 나한테 관심 있다고 하면 어떻게 해? ',
                '남친 사귀는 팁좀줘라ㅠ ',
                '남친 사귀는 팁좀! 나 진짜 연애경험 없는데., 다들 어떻게 하는거야? ',
              ]}
            />
          </div>
        </Scene>
        <Scene pin>
          <div className="section s4">
            <GeneratedResult
              query='일본이랑 중국은'
              results={[
                '일본이랑 중국은 왜 이렇게 서로 까대냐 진짜',
                '일본이랑 중국은 진짜 다르지. 한국에서도 미국에선 한국인들한테 인기 많고, 특히나 여자들이 말하는 남자아이돌 중 하나가 일본인이라면 더 매력적으로 보이기도 함 ',
                '일본이랑 중국은 왜싸우고 미국만 욕함?'
              ]}
            />
          </div>
        </Scene>
        <Scene pin>
          <div className="section s5">
            <GeneratedResult
              query='내가 좋아하는 거는'
              keywords='식사'
              results={[
                '내가 좋아하는 거는 점심 저녁 먹고 운동하기. 밥은 그냥 적당히 챙겨먹는데 살이 안 빠지네',
                '내가 좋아하는 거는 밥 먹고 나서도 항상 점심 때까지 같이 있어주면 좋은 것 같아. 근데 그 시간에 나랑 밥먹을 사람이 없어서 슬프다',
                '내가 좋아하는 거는 밥이랑 술! 근데 난 먹고싶은 음식을 못먹음. 그래서 힘들다 요즘. '
              ]}
            />
          </div>
        </Scene>
        <Scene pin>
          <div className="section s6">
            <GeneratedResult
              query='요즘 밤마다 무슨 생각'
              keywords='사랑, 설렘'
              results={[
                '요즘 밤마다 무슨 생각이 드는데, 나는 항상 그 사람을 좋아했던 거였어. 근데 이 모든 걸 다 놓아버리지 못하겠어ㅠㅠ 내 마음은 이미 떠난 것 같아서 너무 힘들다ㅠ',
                '요즘 밤마다 무슨 생각이 드냐. 그냥 문득문득 생각나서 잠을 잘 수가 없네',
                '요즘 밤마다 무슨 생각이 드는건지 모르겠다. 나를 좋아하던 사람은 날 좋아해주지 않았고, 내가 마음을 열었으면 했지만 이젠 그 친구에게도 상처만 주는 것 같다'
              ]}
            />
          </div>
        </Scene>

        <Scene pin>
          <div className='section s7'>
            <h1>AI 게시글 생성하러 가기</h1>
            <div style={{width: '10px'}}/>
            <Button
              variant='contained'
              onClick={() => navigate('/gptGenerate')}
            >
              이동
            </Button>
          </div>
        </Scene>
      </Controller>

    </div>
  );
}

function GeneratedResult({query, keywords, results=[]}) {
  const renderKeywords = () => {
    if (keywords == undefined) {
      return null;
    }
    return <h1>키워드: {keywords}</h1>;
  };
  return (
    <div className='generated-result'>
      <h1>문맥: {query}</h1>
      {renderKeywords()}
      {results.map((result, idx) => {
        var rest = result.split(query);
        rest = rest.slice(-1).pop();
        return (
          <div key={idx} className='generated'>
            <span className='bold'>AI: </span>
            <div style={{width: '10px'}}/>
            <div className='expanded'>
              <span><span className='bold'>{query}</span> {rest}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}