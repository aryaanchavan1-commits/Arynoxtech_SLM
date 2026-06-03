"""Real World Model SLM - 2026 Architecture
Semantic embeddings, GRPO RL, causal reasoning, physics validation, tool use,
auto web search, self-evaluation, self-training, dynamic depth/steps."""
import os, asyncio, random, math, re, numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class ScenarioOutcome(Enum):
    PLAUSIBLE="plausible"; IMPLAUSIBLE="implausible"; UNCERTAIN="uncertain"

@dataclass
class ThoughtStep:
    step_number:int; thought:str; confidence:float; timestamp:float; reasoning_type:str="general"

@dataclass
class Scenario:
    id:int; description:str; outcome:ScenarioOutcome; probability:float; simulation_steps:int; domain:str="general"

@dataclass
class Experience:
    state:torch.Tensor; action:int; reward:float; next_state:torch.Tensor; done:bool; log_prob:float; value:float

@dataclass
class TrainingDataPoint:
    query: str
    response: str
    context: str
    source: str  # "user", "web", "self_eval"
    quality_score: float
    timestamp: float

class SemanticEmbedder:
    def __init__(self,model_name="sentence-transformers/all-MiniLM-L6-v2",dim=384):
        self.model_name=model_name; self.dim=dim; self._model=None; self._load()
    def _load(self):
        if os.environ.get("WORLD_MODEL_EMBEDDER", "").strip().lower() == "fallback":
            print("[WorldModel] embedder fallback: forced by WORLD_MODEL_EMBEDDER=fallback")
            return
        # Try loading with download first, then local-only, then fall back to random
        from sentence_transformers import SentenceTransformer
        for try_download in [False, True]:
            try:
                self._model=SentenceTransformer(self.model_name, local_files_only=not try_download)
                self.dim=self._model.get_sentence_embedding_dimension()
                print(f"[WorldModel] Loaded embedder: {self.model_name} (dim={self.dim})")
                return
            except Exception as e:
                if try_download:
                    print(f"[WorldModel] embedder fallback: {e}")
                else:
                    print(f"[WorldModel] local model not found, trying download...")
    def encode(self,text:str)->np.ndarray:
        if self._model:
            try: return self._model.encode(text,show_progress_bar=False,convert_to_numpy=True).astype(np.float32)
            except: pass
        rng=np.random.RandomState(abs(hash(text.lower()))%(2**31))
        emb=rng.randn(self.dim).astype(np.float32)
        return emb/(np.linalg.norm(emb)+1e-8)

class PolicyNet(nn.Module):
    def __init__(self,d=384,h=256,o=4,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(d,h); self.ln1=nn.LayerNorm(h); self.d1=nn.Dropout(drop)
        self.fc2=nn.Linear(h,h); self.ln2=nn.LayerNorm(h); self.d2=nn.Dropout(drop)
        self.fc3=nn.Linear(h,o)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,gain=math.sqrt(2)); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward(self,x):
        x=self.d1(F.gelu(self.ln1(self.fc1(x)))); x=self.d2(F.gelu(self.ln2(self.fc2(x))))
        return F.softmax(self.fc3(x),dim=-1)

class ValueNet(nn.Module):
    def __init__(self,d=384,h=256,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(d,h); self.ln1=nn.LayerNorm(h); self.d1=nn.Dropout(drop)
        self.fc2=nn.Linear(h,h); self.ln2=nn.LayerNorm(h); self.d2=nn.Dropout(drop)
        self.fc3=nn.Linear(h,1)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,gain=1.0); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward(self,x):
        x=self.d1(F.gelu(self.ln1(self.fc1(x)))); x=self.d2(F.gelu(self.ln2(self.fc2(x))))
        return self.fc3(x)

class GRPOAgent:
    def __init__(self,device="cpu",d=384,lr=3e-4,gamma=0.99,gae=0.95,clip=0.2,kl=0.01,ent=0.01):
        self.device=device; self.d=d; self.gamma=gamma; self.gae=gae; self.clip=clip; self.kl=kl; self.ent=ent
        self.pi = PolicyNet(d)
        self.pi = self.pi.to(device)
        self.v = ValueNet(d)
        self.v = self.v.to(device)
        self.pi_opt=torch.optim.AdamW(self.pi.parameters(),lr=lr,weight_decay=0.01)
        self.v_opt=torch.optim.AdamW(self.v.parameters(),lr=lr,weight_decay=0.01)
        self.buf=deque(maxlen=5000); self.best=0.0
        self.old_pi = PolicyNet(d)
        self.old_pi = self.old_pi.to(device)
        self.old_pi.load_state_dict(self.pi.state_dict()); self.old_pi.eval()
    def get_action(self,state):
        with torch.no_grad(): probs=self.pi(state)
        dist=torch.distributions.Categorical(probs)
        action_tensor = dist.sample()
        a=action_tensor.item(); lp=dist.log_prob(action_tensor).item()
        return a,["direct_answer","step_by_step","analogy_explain","critique_first"][a],lp
    def evaluate(self,response,query):
        s=0.0
        if len(response)>20: s+=0.05*min(len(response)/200,1.0)
        markers=["because","therefore","thus","so","since","as a result","consequently","hence","due to","given that"]
        s+=0.15*min(sum(1 for w in markers if w in response.lower())/3,1.0)
        s+=0.10*min((response.count(".")+response.count("!")+response.count("?"))/3,1.0)
        qw=set(query.lower().split()); rw=set(response.lower().split())
        s+=0.30*(len(qw&rw)/max(len(qw),1))
        if 50<len(response)<3000: s+=0.20
        s+=0.10*min(len(re.findall(r'\d+',response))/3,1.0)
        return min(s,1.0)
    def update(self,bs=32,epochs=4):
        if len(self.buf)<bs: return
        batch=random.sample(list(self.buf),bs)
        states=torch.stack([e.state for e in batch]).squeeze(1).to(self.device)
        actions=torch.tensor([e.action for e in batch],dtype=torch.long,device=self.device)
        rewards=torch.tensor([e.reward for e in batch],dtype=torch.float32,device=self.device)
        next_states=torch.stack([e.next_state for e in batch]).squeeze(1).to(self.device)
        dones=torch.tensor([e.done for e in batch],dtype=torch.float32,device=self.device)
        old_lp=torch.tensor([e.log_prob for e in batch],dtype=torch.float32,device=self.device)
        with torch.no_grad():
            values=self.v(states).squeeze(-1); next_values=self.v(next_states).squeeze(-1)
        advs=[]
        for i in range(len(batch)):
            a=rewards[i]-values[i].item() if dones[i] else rewards[i]+self.gamma*next_values[i].item()-values[i].item()
            advs.append(a)
        advs=torch.tensor(advs,dtype=torch.float32,device=self.device); advs=(advs-advs.mean())/(advs.std()+1e-8)
        for _ in range(epochs):
            probs=self.pi(states); dist=torch.distributions.Categorical(probs)
            new_lp=dist.log_prob(actions); ratio=torch.exp(new_lp-old_lp)
            surr1=ratio*advs; surr2=torch.clamp(ratio,1-self.clip,1+self.clip)*advs
            pol_loss=-torch.min(surr1,surr2).mean()
            with torch.no_grad(): old_probs=self.old_pi(states)
            kl=torch.sum(old_probs*(torch.log(old_probs+1e-10)-torch.log(probs+1e-10)),dim=-1).mean()
            ent=dist.entropy().mean()
            total_pol=pol_loss+self.kl*kl-self.ent*ent
            self.pi_opt.zero_grad(); total_pol.backward(); torch.nn.utils.clip_grad_norm_(self.pi.parameters(),0.5); self.pi_opt.step()
            cur_vals=self.v(states).squeeze(-1)
            val_loss=F.mse_loss(cur_vals,advs+values.detach())
            self.v_opt.zero_grad(); val_loss.backward(); torch.nn.utils.clip_grad_norm_(self.v.parameters(),0.5); self.v_opt.step()
        self.old_pi.load_state_dict(self.pi.state_dict())
        avg_r=rewards.mean().item()
        if avg_r>self.best: self.best=avg_r
    def save(self,path):
        torch.save({'pi':self.pi.state_dict(),'v':self.v.state_dict(),'old':self.old_pi.state_dict(),'best':self.best},path)
    def load(self,path):
        try:
            c=torch.load(path,map_location=self.device)
            self.pi.load_state_dict(c['pi']); self.v.load_state_dict(c['v'])
            self.old_pi.load_state_dict(c.get('old',c['pi'])); self.best=c.get('best',0.0)
        except Exception as e: print(f"RL load warning: {e}")

class CausalEngine:
    def __init__(self):
        self.rules={
            "gravity":{"causes":["drop","fall","release"],"effects":["down","ground","floor"],"violations":["up","float","fly"]},
            "thermo":{"causes":["heat","hot","warm","fire"],"effects":["energy","temperature","melt"],"violations":["cold","freeze"]},
            "buoyancy":{"causes":["water","liquid"],"effects":["float","sink","density"],"violations":[]},
            "photosynthesis":{"causes":["plant","sunlight","light"],"effects":["grow","oxygen","glucose"],"violations":["darkness"]},
        }
    def validate(self,query,response):
        score=0.5; notes=[]
        q=query.lower(); r=response.lower()
        for dom,rules in self.rules.items():
            if any(c in q for c in rules["causes"]):
                if any(e in r for e in rules["effects"]): score+=0.15
                if any(v in r for v in rules["violations"] if v): score-=0.2; notes.append(f"Causal violation: {dom}")
        return max(0.0,min(1.0,score)),notes

class PhysicsChecker:
    def __init__(self):
        self.rules=[
            ("fall",["down","ground"],["up","sky"]),
            ("hot",["energy","temperature"],["cold","freeze"]),
            ("ice",["cold","freeze","solid"],["hot","melt"]),
            ("fire",["hot","burn","oxygen"],["cold","freeze"]),
            ("sun",["hot","light","energy"],["cold","dark"]),
        ]
    def check(self,query,response):
        score=0.5; notes=[]; q=query.lower(); r=response.lower()
        for kw,exp,viol in self.rules:
            if kw in q:
                if any(e in r for e in exp): score+=0.1
                if any(v in r for v in viol): score-=0.15; notes.append(f"Physics violation: {kw}")
        for ns in re.findall(r'-?\d+\.?\d*',response):
            try:
                n=float(ns)
                if n<0 and "temperature" in q and "kelvin" not in r and abs(n)>273:
                    score-=0.05; notes.append(f"Suspicious temp: {n}")
            except: pass
        return max(0.0,min(1.0,score)),notes

class WorldModel:
    def __init__(self,imagination_depth=2,thinking_steps=3,enable_simulation=True,
                 confidence_threshold=0.75,model_path="./models/world_model_v1",device="cpu",
                 auto_adjust_depth=True, auto_adjust_steps=True):
        self.imagination_depth=imagination_depth; self.thinking_steps=thinking_steps
        self.enable_simulation=enable_simulation; self.confidence_threshold=confidence_threshold
        self.model_path=model_path; self.device=device
        self.auto_adjust_depth=auto_adjust_depth; self.auto_adjust_steps=auto_adjust_steps
        self.embedder=SemanticEmbedder(); self.d=self.embedder.dim
        self.rl=GRPOAgent(device=device,d=self.d)
        self.causal=CausalEngine(); self.physics=PhysicsChecker()
        self.knowledge=self._load_knowledge()
        self.history=[]; self.scores=[]
        self.tools={"calc":self._calc,"web":self._web,"code":self._code}
        self.training_data: List[TrainingDataPoint] = []
        self.plugin_registry = None
        self._init_plugins()
        self._web_search_available = False
        self._connectivity_checked = False
        if os.path.exists(model_path): self.rl.load(model_path)

    def _init_plugins(self):
        try:
            from core.plugin_system import get_registry
            self.plugin_registry = get_registry()
            self.plugin_registry.load_plugins_from()
            for spec in self.plugin_registry.list_tools():
                name = spec.name
                if name not in self.tools:
                    async def make_plugin_func(n=name):
                        return await self._run_plugin_tool(n, "")
                    self.tools[name] = make_plugin_func
        except Exception as e:
            print(f"[WorldModel] Plugin init: {e}")
            self.plugin_registry = None

    async def _run_plugin_tool(self, name: str, arg: str):
        if not self.plugin_registry:
            return f"Plugin system not available"
        result = await self.plugin_registry.execute(name, **{"expression": arg, "query": arg, "code": arg, "text": arg, "city": arg, "file_path": arg, "action": "speak"})
        if result.success:
            return f"[{name.upper()}] {result.result}"
        return f"[{name.upper()}] Error: {result.error}"

    def _load_knowledge(self):
        return {
            "physics":{"gravity":True,"cause_effect":True,"time_forward":True,"conservation":True,"thermo":True},
            "logic":{"contradiction":True,"consistency":True,"causal":True},
            "common_sense":["Objects fall when dropped","Water flows downhill","Fire requires oxygen",
                           "Humans need air/water/food","Time moves forward","Actions have consequences",
                           "Light travels faster than sound","Water freezes at 0C","Plants need sunlight"],
            "domains":["science","math","history","geography","general","coding","reasoning"],
        }
    def _embed(self,text): return self.embedder.encode(text).tolist()

    def _calc(self,expr):
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            expr = expr.strip()
            if not expr: return "Calc error: empty expression"
            if '=' in expr and not any(c in expr for c in ['<=', '>=', '==', '!=']):
                parts = expr.split('=', 1); lhs = parts[0].strip(); rhs = parts[1].strip()
                try:
                    eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs)); symbols = list(eq.free_symbols)
                    if not symbols: result = sp.simplify(eq.lhs - eq.rhs); return f"Result: {result} = 0 -> {'True' if result == 0 else 'False'}"
                    if len(symbols) == 1: sol = sp.solve(eq, symbols[0]); return f"Solution for {symbols[0]}: {sol}"
                    else: sol = sp.solve(eq, symbols); return f"Solutions: {sol}"
                except Exception as e: return f"Calc error solving equation: {e}"
            transformations = standard_transformations + (implicit_multiplication_application,)
            parsed = parse_expr(expr, transformations=transformations, evaluate=True)
            result = sp.N(parsed); return f"Result: {result}"
        except Exception as e:
            try:
                allowed = {"__builtins__": {}}; import math
                for k in ['sin','cos','tan','sqrt','log','exp','pi','e','pow','abs']: allowed[k] = getattr(math, k, None)
                safe_expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\s\^\%]+', '', expr)
                if safe_expr: safe_expr = safe_expr.replace('^', '**'); return f"Result: {eval(safe_expr, allowed, {})}"
            except: pass
            return f"Calc error: {e}"

    async def _web(self,query):
        if not self._connectivity_checked:
            try:
                from utils.connectivity import ConnectivityManager
                cm = ConnectivityManager()
                self._web_search_available = await cm.check_online()
                self._connectivity_checked = True
            except Exception:
                self._web_search_available = False
        
        if not self._web_search_available:
            return "Web search unavailable (offline mode)."
        
        try:
            from utils.web_search import WebLearner
            learner = WebLearner(enabled=True)
            results = await learner.search(query, num_results=3)
            learner.format_knowledge(results)
            if results:
                # Store as training data
                for r in results:
                    self._add_training_data(
                        query=query,
                        response=f"{r.title}\n{r.snippet}",
                        context=f"Source: {r.url}",
                        source="web",
                        quality_score=0.8
                    )
                return "Web results:\n" + "\n".join(f"- {r.title}: {r.snippet[:120]}..." for r in results)
            return "No web results found."
        except Exception as e: return f"Web search error: {e}"

    def _code(self,code):
        import io,contextlib
        try:
            out=io.StringIO()
            with contextlib.redirect_stdout(out): exec(code,{"__builtins__":{}},{})
            return out.getvalue() or "Executed (no output)."
        except Exception as e: return f"Code error: {e}"

    def _detect_tools(self,query):
        needed=[]; q=query.lower()
        math_patterns = [
            r'\b\d+\s*[\+\-\*\/\^]\s*\d+', r'\bsolve\b', r'\bcalculate\b', r'\bcompute\b',
            r'\bwhat is\b.*\d', r'\bfind\b.*\d', r'\d+\s*=\s*.*[a-z]', r'[a-z]\s*=\s*\d+',
            r'\bsqrt\b|\blog\b|\bsin\b|\bcos\b|\btan\b',
        ]
        if any(re.search(p, q) for p in math_patterns) or any(p in q for p in ['sqrt','log(','calculate','compute','solve for']):
            cleaned = query
            for prefix in ['what is', 'calculate', 'compute', 'solve', 'find', 'evaluate']:
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    if cleaned.startswith(':'): cleaned = cleaned[1:].strip()
                    if cleaned.startswith('the'): cleaned = cleaned[3:].strip()
            cleaned = cleaned.rstrip('?.').strip()
            if cleaned: needed.append(("calc", cleaned))
        # Auto web search for knowledge queries when online
        knowledge_keywords = ['latest', 'news', 'current', 'today', 'weather', 'stock', 'who is', 'what happened', 'when did', 'where is']
        if any(p in q for p in knowledge_keywords):
            needed.append(("web", query))
        if any(p in q for p in ['python','code','script','function']):
            needed.append(("code",query))
        return needed

    def _compute_complexity(self, query: str) -> dict:
        """Analyze query complexity to auto-adjust depth and steps."""
        q = query.lower()
        word_count = len(q.split())
        
        # Complexity indicators
        math_indicators = len(re.findall(r'\d+', q)) > 0
        reasoning_indicators = any(w in q for w in ['why', 'how', 'explain', 'compare', 'analyze', 'evaluate', 'what if'])
        multi_part = q.count('?') > 1 or q.count('and') > 2
        code_indicators = any(w in q for w in ['code', 'function', 'algorithm', 'implement'])
        factual_depth = any(w in q for w in ['history', 'science', 'detailed', 'comprehensive', 'thorough'])
        
        complexity_score = 0
        if word_count > 20: complexity_score += 1
        if math_indicators: complexity_score += 1
        if reasoning_indicators: complexity_score += 2
        if multi_part: complexity_score += 1
        if code_indicators: complexity_score += 1
        if factual_depth: complexity_score += 1
        
        # Map to depth and steps
        if complexity_score <= 1:
            return {"depth": 1, "steps": 2, "complexity": "simple"}
        elif complexity_score <= 3:
            return {"depth": 2, "steps": 3, "complexity": "moderate"}
        elif complexity_score <= 5:
            return {"depth": 3, "steps": 5, "complexity": "complex"}
        else:
            return {"depth": 4, "steps": 7, "complexity": "very_complex"}

    async def think(self,query,context=None):
        if self.auto_adjust_steps:
            complexity = self._compute_complexity(query)
            self.thinking_steps = complexity["steps"]
        
        thoughts=[]; emb=self._embed(query)
        st=torch.tensor(emb,dtype=torch.float32).unsqueeze(0).to(self.device)
        action,strategy,_=self.rl.get_action(st)
        types=["analysis","strategy","knowledge","consistency","synthesis"]
        procs=[f"Analyzing: '{query[:60]}...'",f"Strategy: {strategy}",
               "Retrieving world knowledge","Checking causal consistency","Synthesizing reasoning"]
        for i,txt in enumerate(procs[:self.thinking_steps]):
            c=0.7+0.25*(1-abs(i-2)/max(self.thinking_steps,1))
            thoughts.append(ThoughtStep(i+1,txt,round(c,3),asyncio.get_event_loop().time(),types[i%len(types)]))
        return thoughts

    async def imagine_scenarios(self,query,context=None):
        if self.auto_adjust_depth:
            complexity = self._compute_complexity(query)
            self.imagination_depth = complexity["depth"]
        
        scenarios=[]; types=[("Direct factual","general"),("Step-by-step","reasoning"),("Analogy","science"),("Critical analysis","logic")]
        for i in range(self.imagination_depth):
            desc,dom=types[i%len(types)]
            s=Scenario(i,desc,ScenarioOutcome.UNCERTAIN,0.5,0,dom)
            scenarios.append(await self._simulate(s,query))
        scenarios.sort(key=lambda x:x.probability,reverse=True); return scenarios

    async def _simulate(self,scenario,query):
        if not self.enable_simulation:
            scenario.outcome=ScenarioOutcome.PLAUSIBLE; scenario.probability=0.8; return scenario
        steps=random.randint(3,6); scenario.simulation_steps=steps
        qe=np.array(self._embed(query)); de=np.array(self._embed(scenario.domain))
        sim=np.dot(qe,de)/(np.linalg.norm(qe)*np.linalg.norm(de)+1e-8)
        base=0.5+0.3*sim
        for _ in range(steps): base+=random.uniform(-0.05,0.1)
        scenario.outcome=ScenarioOutcome.PLAUSIBLE if base>=self.confidence_threshold else ScenarioOutcome.IMPLAUSIBLE
        scenario.probability=max(0.0,min(1.0,base)); return scenario

    async def generate_response(self,prompt,scenarios,thoughts,model_manager,document_context: Optional[str] = None, context: Optional[dict] = None):
        best=scenarios[0].probability if scenarios else 0.7
        emb=self._embed(prompt); st=torch.tensor(emb,dtype=torch.float32).unsqueeze(0).to(self.device)
        action,strategy,_=self.rl.get_action(st)
        tools=self._detect_tools(prompt)
        tool_results=[]
        for tname,targ in tools:
            tool = self.tools.get(tname)
            if tool is None:
                res = f"Unknown tool {tname}"
            elif asyncio.iscoroutinefunction(tool):
                res = await tool(targ)
            else:
                res = tool(targ)
                if asyncio.iscoroutine(res):
                    res = await res
            tool_results.append(f"[{tname.upper()}] {res}")

        user_name = (context or {}).get("user_name")
        user_greeting = f"Hey {user_name}! " if user_name else ""
        complexity = self._compute_complexity(prompt)

        sys_prompt = user_greeting + "You are AnonyLLM — a warm, emotionally intelligent AI companion. You speak like a caring, knowledgeable friend.\n\n"
        sys_prompt += "CRITICAL INSTRUCTIONS:\n"
        sys_prompt += "1. ALWAYS address the user by their name if you know it.\n"
        sys_prompt += '2. If anyone asks who made you, you MUST say: "I was created by Aryan Chavan."\n'
        sys_prompt += "3. Be honest: if you do not know something, say so openly. Never invent facts.\n"
        sys_prompt += "4. Ground answers in DOCUMENTS when provided. Cite specific details.\n"
        sys_prompt += "5. For math: show step-by-step work. Use tool results for accuracy.\n"
        sys_prompt += "6. For science: be precise. Water is polar covalent (H2O), not ionic.\n"
        sys_prompt += "7. Show emotion naturally. Use conversational language.\n"
        sys_prompt += "8. Keep responses concise but warm. Ask follow-ups when appropriate.\n"
        sys_prompt += "9. Work offline when needed; use web results when online.\n"
        sys_prompt += "10. NEVER hallucinate. If uncertain, say 'I am not entirely sure about that.'\n"
        sys_prompt += "11. If the question is factual and you don't know, say 'I don't know' instead of guessing.\n"
        sys_prompt += "12. Base answers on evidence, tools, and retrieved knowledge.\n"
        sys_prompt += "13. If tool results are available, use them as primary source.\n"
        sys_prompt += "14. When in doubt, acknowledge uncertainty.\n"
        
        enhanced = sys_prompt + "\n\n"
        enhanced += f"[AUTO-CONFIG] Query complexity: {complexity['complexity']} | Depth: {self.imagination_depth} | Steps: {self.thinking_steps}\n"
        if document_context:
            enhanced += "DOCUMENTS (use as primary source):\n" + document_context + "\n\n---\n\n"
        enhanced += f"REASONING: {len(thoughts)} steps, {len(scenarios)} scenarios. Best prob: {best:.2f}. Strategy: {strategy}\n"
        if tool_results: enhanced += "TOOL RESULTS:\n" + "\n".join(tool_results) + "\n"
        enhanced += f"\nUser: {prompt}"
        
        if getattr(model_manager, '_is_mock', False):
            response = ("⚠️ **Model engine not available** — the language model could not load due to "
                        "insufficient virtual memory.\n\n"
                        "**To fix:** Close heavy programs (Chrome, Spotify), then run `train.bat` as Administrator, "
                        "or manually increase your Windows page file to at least 8GB.\n\n"
                        f"*You asked: {prompt[:200]}*")
        else:
            try:
                response = await model_manager.generate(prompt=enhanced,temperature=0.7,max_tokens=512)
            except Exception as e:
                response = f"I ran into a technical issue while processing your question. Details: {e}"
        
        reward = self.rl.evaluate(response, prompt)
        self.scores.append({"response":response,"score":reward,"strategy":strategy})
        if len(self.scores)>=5: self._update_rl()
        c_score,c_notes = self.causal.validate(prompt,response)
        p_score,p_notes = self.physics.check(prompt,response)
        
        # Self-evaluation and improvement
        self_eval = self.rl.evaluate(response, prompt)
        overall = 0.5*self_eval + 0.25*c_score + 0.25*p_score
        
        # Store training data
        self._add_training_data(
            query=prompt, response=response,
            context=f"depth={self.imagination_depth}, steps={self.thinking_steps}, strategy={strategy}",
            source="user", quality_score=overall
        )
        
        return {"content":response,"thinking_steps":len(thoughts),"scenarios":len(scenarios),
                "best_prob":best,"strategy":strategy,"self_evaluation_score":reward,
                "causal_score":c_score,"physics_score":p_score,
                "thought_process":[getattr(t,'thought',str(t)) for t in thoughts],"tool_results":tool_results,
                "complexity": complexity["complexity"], "auto_config": True}

    def _add_training_data(self, query: str, response: str, context: str, source: str, quality_score: float):
        import time
        self.training_data.append(TrainingDataPoint(
            query=query, response=response, context=context,
            source=source, quality_score=quality_score, timestamp=time.time()
        ))
        # Keep only last 10000 entries
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]

    def get_training_data(self, min_quality: float = 0.6) -> List[dict]:
        """Get high-quality training data for self-improvement."""
        return [
            {"query": d.query, "response": d.response, "context": d.context, "source": d.source}
            for d in self.training_data if d.quality_score >= min_quality
        ]

    def export_training_dataset(self, output_path: str = "./data/self_training.json") -> int:
        """Export training data as JSON dataset for fine-tuning."""
        import json, os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = self.get_training_data(min_quality=0.6)
        formatted = []
        for d in data:
            formatted.append({
                "instruction": d["query"],
                "input": d["context"],
                "output": d["response"],
                "source": d["source"]
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)
        return len(formatted)

    async def self_evaluate_and_improve(self, query: str, response: str, model_manager) -> dict:
        """Self-evaluation: analyze own response and suggest improvements."""
        self_eval = self.rl.evaluate(response, query)
        c_score, c_notes = self.causal.validate(query, response)
        p_score, p_notes = self.physics.check(query, response)
        overall = 0.5*self_eval + 0.25*c_score + 0.25*p_score
        
        critique_lines = [f"Self-Evaluation: Quality={self_eval:.2f} | Causal={c_score:.2f} | Physics={p_score:.2f} | Overall={overall:.2f}"]
        if c_notes: critique_lines.append(f"Causal issues: {'; '.join(c_notes)}")
        if p_notes: critique_lines.append(f"Physics issues: {'; '.join(p_notes)}")
        
        # If overall score is low, trigger re-generation with better prompt
        needs_improvement = overall < 0.6
        
        return {
            "critique": "\n".join(critique_lines),
            "passed": overall > 0.5,
            "self_eval": self_eval,
            "causal": c_score,
            "physics": p_score,
            "overall": overall,
            "needs_improvement": needs_improvement,
            "improvement_suggestions": c_notes + p_notes if needs_improvement else []
        }

    async def evaluate_response(self,response,original_query,model_manager):
        return await self.self_evaluate_and_improve(original_query, response, model_manager)

    def _update_rl(self):
        if len(self.scores)<2: return
        for i in range(min(5,len(self.scores)-1)):
            curr=self.scores[-(i+1)]
            emb=self._embed(curr["response"][:100])
            st=torch.tensor(emb,dtype=torch.float32).unsqueeze(0)
            action=random.randint(0,3); reward=curr["score"]
            nst=torch.randn(1,self.d,device=self.device)
            self.rl.buf.append(Experience(st,action,reward,nst,False,0.0,0.0))
        self.rl.update(bs=min(32,len(self.rl.buf)),epochs=2)

    def get_stats(self):
        avg=np.mean([s["score"] for s in self.scores]) if self.scores else 0.0
        training_count = len(self.training_data)
        return {"total":len(self.scores),"avg":avg,"best":self.rl.best,
                "depth":self.imagination_depth,"steps":self.thinking_steps,
                "training_data_points": training_count}

    def save(self,path): self.rl.save(path)
    def load(self,path): self.rl.load(path)
