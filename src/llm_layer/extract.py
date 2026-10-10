"""Rule-based implementation of the callback-time-extraction skill for English call summaries."""
import re
import datetime as dt

NUM = {'one': 1, 'a': 1, 'an': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
       'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40,
       'forty-five': 45, 'couple of': 2, 'a couple of': 2, 'few': 3, 'a few': 3, 'half an': 0.5}
WD = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
MON = {m: i for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                    'september', 'october', 'november', 'december'], 1)}
FEST = {'diwali': dt.date(2026, 11, 9), 'deepawali': dt.date(2026, 11, 9), 'dussehra': dt.date(2026, 10, 21),
        'dasara': dt.date(2026, 10, 21), 'navratri': dt.date(2026, 10, 21), 'chhath': dt.date(2026, 11, 17)}
SELLER = r"\b(lead|seller|customer|client|owner|he|she|they|person|caller|proprietor|user)\b"
REQ = (r"(ask|request|said|say|state|told|mention|suggest|prefer|want|instruct|will be|would be|could be|"
       r"is available|are available|be free|be available|call (him |her |them )?back|callback|call later|call again|"
       r"reach out|contact|agree|inform|indicat|promis|plan|busy|unavailable|not available|travel|out of station|"
       r"unwell|sick|hospital|occupied)")
WRONG = (r"(wrong number|not (from|associated with|related to|part of|the owner of|connected to) (that |the |this )?"
         r"(company|firm|business|organi[sz]ation)|no such (company|business|person)|"
         r"not the (right|correct|intended) (person|number|contact)|incorrect number|dialed the wrong|misdial)")
DNC = (r"(not to call (him |her |them )?again|stop calling|do not call|don't call|never call|"
       r"remove (his|her|their) (number|contact)|remove .{0,20}from (the )?(calling|contact) list|"
       r"not be contacted|no further calls|unsubscribe)")
VAGUE = (r"(call (him |her |them )?(back )?later|callback later|call back when|when (he|she|they) (is|are|was|were) free|"
         r"busy|not (free|available) (at the moment|right now|now|currently)|later time|some other time|another time|"
         r"will call back|would call back|get back|reach out later|call again later)")


def sentences(t):
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+', t) if x.strip()]


def seller_sentence(s):
    l = s.lower()
    if re.search(r"(was|is|were|has been|have been|got)\s+(re)?(scheduled|fixed|set|planned|booked|confirmed)", l):
        return True
    if re.search(r"\bagent\b", l) and not re.search(SELLER, l):
        return False
    if re.match(r"^the agent (offered|proposed|suggested|asked|tried|attempted|mentioned|informed|explained|called)", l) \
            and not re.search(r"\b(lead|seller|customer|he|she|they)\b[^.]*\b(agree|accept|asked|request|said|told|prefer|suggest|confirm)", l):
        return False
    return bool(re.search(SELLER, l)) and bool(re.search(REQ, l))


def clock(l):
    m = re.search(r"\b(?:at|around|by|after|before|about|between|from)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?|o'?clock)?", l)
    if not m:
        return None
    h = int(m.group(1)); mi = int(m.group(2) or 0); ap = (m.group(3) or '').replace('.', '')
    if not ap and m.group(2) is None:
        return None  # a bare number ("after 2") is too ambiguous
    if h > 23 or mi > 59:
        return None
    if ap.startswith('p') and h < 12:
        h += 12
    elif ap.startswith('a') and h == 12:
        h = 0
    elif (not ap or ap.startswith('o')) and 1 <= h <= 8:
        h += 12
    return h, mi


def nextwd(d, w):
    k = (w - d.weekday()) % 7
    return d + dt.timedelta(days=k or 7)


def extract(text, t):
    """text: summary; t: call datetime. Returns the skill's fields (subset)."""
    out = dict(cue_strength='none', cue_type='none', target=None, win_start=None, win_end=None, earliest=None,
               wrong_number=False, dnc=False, evidence='')
    if not isinstance(text, str) or len(text) < 20 or text.strip().upper() == 'NA':
        return out
    text = re.sub(r"[‐-―]", "-", text).replace(" ", " ").replace(" ", " ")
    low = text.lower()
    out['wrong_number'] = bool(re.search(WRONG, low))
    out['dnc'] = bool(re.search(DNC, low))
    for s in sentences(text):
        if not seller_sentence(s):
            continue
        l = s.lower()
        if re.search(r"\b(refus|declin|reject|turned down)|\b(would|will|does|do|did) not (work|suit)\b|\bnot (possible|convenient)\b", l) \
                and not re.search(r"\b(instead|but|however|rather)\b[^.]*\b(asked|request|suggest|prefer|agree|call)", l):
            continue
        day = None; cue = None
        if 'day after tomorrow' in l:
            day = t.date() + dt.timedelta(days=2); cue = 'tomorrow'
        elif re.search(r"\b(tomorrow|next day)\b", l):
            day = t.date() + dt.timedelta(days=1); cue = 'tomorrow'
        last = -1
        for i, w in enumerate(WD):
            for mw in re.finditer(r"\b" + w + r"\b", l):
                if mw.start() > last:
                    last = mw.start(); day = nextwd(t.date(), i); cue = 'weekday'
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(MON) + r")\b|\b(" + "|".join(MON) +
                      r")\s+(\d{1,2})(?:st|nd|rd|th)?\b|\b(?:on|by|around|after)\s+the\s+(\d{1,2})(?:st|nd|rd|th)\b", l)
        if m:
            try:
                if m.group(1):
                    d = dt.date(t.year, MON[m.group(2)], int(m.group(1)))
                elif m.group(3):
                    d = dt.date(t.year, MON[m.group(3)], int(m.group(4)))
                else:
                    dd = int(m.group(5)); d = dt.date(t.year, t.month, dd)
                    if d < t.date() or 'next month' in l:
                        d = dt.date(t.year + (t.month == 12), t.month % 12 + 1, dd)
                if d >= t.date():
                    day = d; cue = 'date'
            except ValueError:
                pass
        mm = re.search(r"\b(?:after|in|within)\s+(\d+|ten|fifteen|twenty|thirty|forty-five|forty|five|two|three|four|"
                       r"half an|a couple of|couple of|a few|few)\s*(?:(?:-|to)\s*\d+\s*)?(minutes?|mins?|hours?|hrs?|days?|weeks?)\b", l)
        if mm and cue not in ('date', 'weekday'):
            n = mm.group(1); n = float(n) if n.isdigit() else NUM.get(n, 1); unit = mm.group(2)
            if unit.startswith('min'):
                out.update(cue_strength='explicit', cue_type='relative_minutes', target=t + dt.timedelta(minutes=n), evidence=s); return out
            if unit.startswith('h'):
                out.update(cue_strength='explicit', cue_type='relative_hours', target=t + dt.timedelta(hours=n), evidence=s); return out
            days = max(1, int(n)) * (7 if unit.startswith('week') else 1)
            out.update(cue_strength='implicit', cue_type='relative_days', earliest=t.date() + dt.timedelta(days=days), evidence=s)
            continue
        if re.search(r"\b(after|in)\s+an hour\b", l):
            out.update(cue_strength='explicit', cue_type='relative_hours', target=t + dt.timedelta(hours=1), evidence=s); return out
        ck = clock(l)
        if ck:
            d = day or t.date(); h, mi = ck
            tgt = dt.datetime.combine(d, dt.time(h, mi))
            if tgt <= t and not day:
                tgt += dt.timedelta(days=1)
            if tgt > t:
                out.update(cue_strength='explicit', cue_type='clock', target=tgt, evidence=s); return out
        dp = None
        neg = lambda w: re.search(r"\bnot\b[^.]{0,25}\b" + w, l)
        if re.search(r"\b(evening|after work|after closing|shop closes)\b", l) and not neg('evening'): dp = (17, 20)
        elif re.search(r"\b(night|tonight)\b", l) and not neg('night'): dp = (19, 21)
        elif re.search(r"\b(afternoon|after lunch|post lunch|lunchtime)\b", l) and not neg('(afternoon|lunch)'): dp = (14, 16)
        elif re.search(r"\bmorning\b", l) and not neg('morning'): dp = (10, 12)
        if day:
            if dp:
                out.update(cue_strength='implicit', cue_type=cue, win_start=dt.datetime.combine(day, dt.time(dp[0])),
                           win_end=dt.datetime.combine(day, dt.time(dp[1])), evidence=s)
            else:
                out.update(cue_strength='explicit', cue_type=cue, target=dt.datetime.combine(day, dt.time(11)), evidence=s)
            return out
        if dp:
            d = t.date() if t.hour < dp[1] else t.date() + dt.timedelta(days=1)
            out.update(cue_strength='implicit', cue_type='daypart', win_start=dt.datetime.combine(d, dt.time(dp[0])),
                       win_end=dt.datetime.combine(d, dt.time(dp[1])), evidence=s)
            continue
        if re.search(r"\bnext week\b", l):
            out.update(cue_strength='implicit', cue_type='relative_days', earliest=nextwd(t.date(), 0), evidence=s); continue
        if re.search(r"\bnext month\b", l):
            out.update(cue_strength='implicit', cue_type='relative_days',
                       earliest=dt.date(t.year + (t.month == 12), t.month % 12 + 1, 1), evidence=s); continue
        hit = False
        for f, fd in FEST.items():
            if re.search(r"\b(after|post)\b[^.]{0,25}\b" + f, l) or (f in l and 'festiv' in l):
                if fd > t.date():
                    out.update(cue_strength='implicit', cue_type='event', earliest=fd, evidence=s); hit = True
                break
        if hit:
            continue
        if re.search(r"\btravel(l)?ing\b|\bout of (station|town|the city)\b|\bnot in (town|the city|station)\b|\bunwell\b|\bsick\b|\bhospital", l):
            out.update(cue_strength='implicit', cue_type='unavailable_period', earliest=t.date() + dt.timedelta(days=2), evidence=s); continue
        if out['cue_strength'] == 'none' and re.search(VAGUE, l):
            out.update(cue_type='vague', evidence=s)
    return out
