import React from 'react';
import { DebugLogger } from '../utils/utils';
import ScreenWrapper from './ScreenWrapper';
import { nl2br } from '../utils/nl2br';

export default function LogScreen() {
  return (
    <ScreenWrapper>
      <div className="debug-log-screen pt-8">
        <h1 className="text-2xl mb-4">Debug log</h1>

        <div className="font-mono bg-base-200 p-2 rounded-lg text-sm mb-16">
          {DebugLogger.content.length === 0 && (
            <span>(Log is empty. Please firstly load a model)</span>
          )}
          {DebugLogger.content.map((line, i) => (
            <React.Fragment key={i}>
              {nl2br(line)}
              <br />
            </React.Fragment>
          ))}
        </div>
      </div>
    </ScreenWrapper>
  );
}
