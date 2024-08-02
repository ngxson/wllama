import React from 'react';

export function nl2br(text: string) {
  return text.split('\n').map((line, i) => (
    <React.Fragment key={i}>
      {line}
      <br />
    </React.Fragment>
  ));
}
