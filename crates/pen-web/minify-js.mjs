#!/usr/bin/env node
// Zero-dependency JS comment & whitespace minifier for pen-web glue scripts.
// Preserves strings, nested template literals, regex literals, and statement
// newlines (ASI-safe), while stripping JSDoc/line comments, indentation,
// blank lines, and redundant horizontal spaces around punctuators.
import fs from 'node:fs';

const REGEX_PREV_WORDS = new Set([
  'return', 'throw', 'case', 'delete', 'void', 'typeof', 'in', 'instanceof',
  'new', 'of', 'yield', 'await', 'else', 'do', 'export', 'default',
]);
const REGEX_PREV_PUNCT = new Set([
  '(', '[', '{', ',', ';', ':', '?', '=', '!', '&', '|', '^', '~',
  '+', '-', '*', '/', '%', '<', '>',
]);

function isIdentStart(ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_' || ch === '$' || ch.charCodeAt(0) >= 0x80;
}
function isIdentPart(ch) {
  return isIdentStart(ch) || (ch >= '0' && ch <= '9');
}

function minify(src) {
  const n = src.length;
  const out = [];
  let i = 0;
  let lastTokKind = 'start'; // 'start' | 'ident' | 'number' | 'string' | 'punct' | 'close'
  let lastTokText = '';
  let pendingSpace = false;
  let pendingNewline = false;
  // Stack tracking brace depth inside `${...}` template expressions.
  const tplBraceStack = [];

  function canStartRegex() {
    if (lastTokKind === 'start') return true;
    if (lastTokKind === 'punct') return REGEX_PREV_PUNCT.has(lastTokText);
    if (lastTokKind === 'ident') return REGEX_PREV_WORDS.has(lastTokText);
    return false;
  }

  function needsSpace(prevChar, nextChar, nextKind) {
    if (!prevChar) return false;
    if (isIdentPart(prevChar) && (isIdentPart(nextChar) || nextKind === 'number')) return true;
    if (prevChar === '+' && nextChar === '+') return true;
    if (prevChar === '-' && nextChar === '-') return true;
    if (prevChar === '/' && (nextChar === '/' || nextChar === '*')) return true;
    if (prevChar === '<' && nextChar === '!') return true;
    if (prevChar === '>' && nextChar === '=') return true;
    if (prevChar === '=' && nextChar === '=') return true;
    if (prevChar === '!' && nextChar === '=') return true;
    // Preserve space between `in`/`of` and regex/strings or numeric dot ambiguity like `1 .toString()`.
    if (lastTokKind === 'number' && nextChar === '.') return true;
    return false;
  }

  function emit(text, kind, tokText = text) {
    if (out.length > 0) {
      const prev = out[out.length - 1];
      const prevChar = prev[prev.length - 1];
      const nextChar = text[0];
      if (pendingNewline) {
        // Keep statement-boundary newlines so ASI stays 100% intact.
        const safeNoNewline =
          (prevChar === '{' || prevChar === ';' || prevChar === ',' || prevChar === '(' || prevChar === '[') &&
          nextChar !== '/';
        if (!safeNoNewline) {
          out.push('\n');
        } else if (needsSpace(prevChar, nextChar, kind)) {
          out.push(' ');
        }
      } else if (pendingSpace && needsSpace(prevChar, nextChar, kind)) {
        out.push(' ');
      }
    }
    pendingSpace = false;
    pendingNewline = false;
    out.push(text);
    lastTokKind = kind;
    lastTokText = tokText;
  }

  function scanTemplateChunk(startIdx) {
    let j = startIdx;
    let buf = src[j++]; // '`' or '}'
    while (j < n) {
      const c = src[j];
      if (c === '\\') {
        buf += src.slice(j, j + 2);
        j += 2;
      } else if (c === '`') {
        buf += '`';
        j++;
        return { text: buf, next: j, opensExpr: false };
      } else if (c === '$' && src[j + 1] === '{') {
        buf += '${';
        j += 2;
        return { text: buf, next: j, opensExpr: true };
      } else {
        buf += c;
        j++;
      }
    }
    return { text: buf, next: j, opensExpr: false };
  }

  while (i < n) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\r' || c === '\f' || c === '\v') {
      pendingSpace = true;
      i++;
      continue;
    }
    if (c === '\n') {
      pendingNewline = true;
      i++;
      continue;
    }
    // Line comment
    if (c === '/' && src[i + 1] === '/') {
      i += 2;
      while (i < n && src[i] !== '\n') i++;
      continue;
    }
    // Block comment
    if (c === '/' && src[i + 1] === '*') {
      let hasNl = false;
      i += 2;
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) {
        if (src[i] === '\n') hasNl = true;
        i++;
      }
      i += 2;
      if (hasNl) pendingNewline = true;
      else pendingSpace = true;
      continue;
    }
    // Strings
    if (c === '\'' || c === '"') {
      const quote = c;
      let j = i + 1;
      while (j < n && src[j] !== quote) {
        if (src[j] === '\\') j += 2;
        else j++;
      }
      j = Math.min(n, j + 1);
      emit(src.slice(i, j), 'string', quote);
      i = j;
      continue;
    }
    // Template literal start
    if (c === '`') {
      const chunk = scanTemplateChunk(i);
      emit(chunk.text, chunk.opensExpr ? 'punct' : 'string', chunk.opensExpr ? '{' : '`');
      if (chunk.opensExpr) tplBraceStack.push(0);
      i = chunk.next;
      continue;
    }
    // Closing brace of a `${...}` template expression
    if (c === '}' && tplBraceStack.length > 0 && tplBraceStack[tplBraceStack.length - 1] === 0) {
      tplBraceStack.pop();
      const chunk = scanTemplateChunk(i);
      pendingSpace = false;
      pendingNewline = false;
      out.push(chunk.text);
      if (chunk.opensExpr) {
        tplBraceStack.push(0);
        lastTokKind = 'punct';
        lastTokText = '{';
      } else {
        lastTokKind = 'string';
        lastTokText = '`';
      }
      i = chunk.next;
      continue;
    }
    // Regex literal
    if (c === '/' && canStartRegex()) {
      let j = i + 1;
      let inClass = false;
      while (j < n) {
        const rc = src[j];
        if (rc === '\\') {
          j += 2;
          continue;
        }
        if (rc === '[') inClass = true;
        else if (rc === ']') inClass = false;
        else if (rc === '/' && !inClass) {
          j++;
          while (j < n && isIdentStart(src[j])) j++;
          break;
        }
        j++;
      }
      emit(src.slice(i, j), 'string', '/');
      i = j;
      continue;
    }
    // Identifier / keyword
    if (isIdentStart(c)) {
      let j = i + 1;
      while (j < n && isIdentPart(src[j])) j++;
      const word = src.slice(i, j);
      emit(word, 'ident', word);
      i = j;
      continue;
    }
    // Number
    if ((c >= '0' && c <= '9') || (c === '.' && src[i + 1] >= '0' && src[i + 1] <= '9')) {
      let j = i + 1;
      while (j < n && (isIdentPart(src[j]) || src[j] === '.')) {
        if ((src[j] === 'e' || src[j] === 'E') && (src[j + 1] === '+' || src[j + 1] === '-')) {
          j += 2;
          continue;
        }
        j++;
      }
      emit(src.slice(i, j), 'number', '0');
      i = j;
      continue;
    }
    // Multi-char operators that affect regex/spacing
    const two = src.slice(i, i + 2);
    if (two === '++' || two === '--') {
      emit(two, 'close', two);
      i += 2;
      continue;
    }
    if (c === '{') {
      if (tplBraceStack.length > 0) tplBraceStack[tplBraceStack.length - 1]++;
      emit(c, 'punct', c);
      i++;
      continue;
    }
    if (c === '}') {
      if (tplBraceStack.length > 0) tplBraceStack[tplBraceStack.length - 1]--;
      emit(c, 'close', c);
      i++;
      continue;
    }
    if (c === ')' || c === ']') {
      emit(c, 'close', c);
      i++;
      continue;
    }
    emit(c, 'punct', c);
    i++;
  }
  return out.join('') + '\n';
}

for (const file of process.argv.slice(2)) {
  const input = fs.readFileSync(file, 'utf8');
  const output = minify(input);
  fs.writeFileSync(file, output, 'utf8');
}
