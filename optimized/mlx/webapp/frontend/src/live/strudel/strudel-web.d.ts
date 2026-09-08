/**
 * @strudel/web ships no type declarations. We only reach for a handful of its
 * exports, so declare those rather than switching the whole module to `any`.
 */
declare module '@strudel/web' {
  export function initStrudel(options?: Record<string, unknown>): Promise<any>;
  export function getAudioContext(): AudioContext;
  export function getSuperdoughAudioController(): any;
  export function setSuperdoughAudioController(c: any): any;
  export function samples(url: string): Promise<void>;
  export function evaluate(code: string, autoplay?: boolean): Promise<any>;
  export function hush(): void;
  export function setcps(cps: number): void;
  export function setcpm(cpm: number): void;
}
