import React, {useEffect} from 'react';
import Sidebar from 'components/bars/sidebar.js';
import TextGenerator from 'components/text_generator.js';

import 'css/gpt_generate.css';

export default function GptGenerateScreen() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div id='gpt-generate-screen'>
      <Sidebar/>
      <h1>문맥 고려 문장 생성</h1>

      <div style={{height: '30px'}}/>

      <TextGenerator
        showKeywords={false}
        initialText='사실 어제 여친이랑 헤어져서 '
      />

      <div style={{height: '100px'}}/>

      <h1>키워드 기반 문장 생성</h1>

      <div style={{height: '30px'}}/>

      <TextGenerator
        initialText='우리나라는 진짜'
      />
    </div>
  );
}