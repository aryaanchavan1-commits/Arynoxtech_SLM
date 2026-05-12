import asyncio
import os
import sys
import tempfile

sys.modules['tensorflow'] = None
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=4)

sys.path.insert(0, str(os.path.dirname(__file__) + "/.."))
from agents import GeneratorAgent, CriticAgent, PromptOptimizerAgent, PlannerAgent, FeedbackLoop
from agents.config.settings import Settings
from memory import MemoryManager, create_vector_store
from utils.logger import get_logger, setup_logging
from utils.connectivity import ConnectivityManager
from core.world_model import WorldModel
from serving.model import ModelManager
from ui.auth import (
    authenticate_user, register_user, load_user_data, save_user_data,
    export_user_data, delete_user_account, user_exists
)

logger = get_logger(__name__)
setup_logging(level="INFO")


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            return _executor.submit(asyncio.run, coro).result()


def init_session_state():
    defaults = {
        "messages": [],
        "world_model": lambda: WorldModel(auto_adjust_depth=True, auto_adjust_steps=True),
"model_manager": lambda: ModelManager(model_path="./models/nemotron-slm-final"),
        "memory_manager": None,
        "initialized": False,
        "uploaded_files": [],
        "user_name": None,
        "user_info": None,
        "logged_in": False,
        "login_tab": "login",
        "online_status": None,
        "last_online_check": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val() if callable(val) else val


def _load_user_session(username: str):
    """Load user's saved chat history and files."""
    data = load_user_data(username)
    st.session_state.messages = data.get("messages", [])
    st.session_state.uploaded_files = data.get("uploaded_files", [])
    saved_settings = data.get("settings", {})
    if saved_settings.get("world_model_depth"):
        st.session_state.world_model.imagination_depth = saved_settings["world_model_depth"]
    if saved_settings.get("world_model_steps"):
        st.session_state.world_model.thinking_steps = saved_settings["world_model_steps"]


def _save_user_session(username: str):
    """Save current session to user's data file."""
    data = {
        "messages": st.session_state.messages,
        "uploaded_files": st.session_state.uploaded_files,
        "settings": {
            "world_model_depth": st.session_state.world_model.imagination_depth,
            "world_model_steps": st.session_state.world_model.thinking_steps,
        }
    }
    save_user_data(username, data)


@st.cache_resource
def get_memory_manager():
    settings = Settings()
    try:
        vs = create_vector_store(
            backend=settings.memory.vector_db,
            embedding_model=settings.memory.embedding_model,
            persist_directory=settings.memory.persist_directory,
        )
        return MemoryManager(
            vector_store=vs,
            persist_directory=settings.memory.persist_directory,
            chunk_size=settings.document_processing.chunk_size,
            chunk_overlap=settings.document_processing.chunk_overlap,
        )
    except Exception as e:
        logger.warning(f"Memory init failed: {e}")
        return None


def check_online_status():
    """Check and cache online status."""
    import time
    now = time.time()
    if st.session_state.online_status is None or (now - st.session_state.last_online_check) > 30:
        cm = ConnectivityManager()
        st.session_state.online_status = cm.check_online_sync()
        st.session_state.last_online_check = now
    return st.session_state.online_status


def process_uploaded_file(uploaded_file, memory_manager):
    if memory_manager is None:
        return {"success": False, "error": "Memory manager not available"}
    try:
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        suffix = os.path.splitext(file_name)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        result = memory_manager.process_document(file_path=tmp_path, file_name=file_name, file_type=file_type)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return result
    except Exception as e:
        logger.error(f"File upload processing failed: {e}")
        return {"success": False, "error": str(e)}


def auth_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1>🌍 World Model SLM 2026</h1>
                <p style="font-size: 1.2rem; color: #666;">Your personal AI companion</p>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        tab_login, tab_register = st.tabs(["🔑 Log In", "📝 Register"])

        with tab_login:
            st.subheader("Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="your_username", max_chars=50)
                password = st.text_input("Password", type="password", placeholder="••••••")
                submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")
                if submitted:
                    if not username.strip() or not password:
                        st.error("Please enter both username and password.")
                    else:
                        success, user_info = authenticate_user(username.strip(), password)
                        if success:
                            st.session_state.user_name = user_info["display_name"]
                            st.session_state.user_info = user_info
                            st.session_state.logged_in = True
                            _load_user_session(user_info["username"])
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        with tab_register:
            st.subheader("Create Account")
            st.write("Sign up to save your chat history and preferences.")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username", placeholder="e.g., aryan2026", max_chars=50, key="reg_user")
                display_name = st.text_input("Display Name", placeholder="e.g., Aryan", max_chars=50, key="reg_name")
                new_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pass")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="••••••", key="reg_confirm")
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submitted:
                    if not new_username.strip() or not new_password:
                        st.error("Username and password are required.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = register_user(new_username.strip(), new_password, display_name.strip() or None)
                        if success:
                            st.success(msg)
                            st.info("Please switch to the Log In tab to sign in.")
                        else:
                            st.error(msg)

        st.divider()
        st.caption("🔒 Works completely offline • Your data stays on your device")


def run_app():
    st.set_page_config(page_title="World Model SLM 2026", page_icon="🌍", layout="wide")
    init_session_state()

    if not st.session_state.logged_in:
        auth_screen()
        return

    user_name = st.session_state.user_name
    username = st.session_state.user_info["username"] if st.session_state.user_info else ""
    is_online = check_online_status()

    status_emoji = "🟢" if is_online else "🔴"
    status_text = "Online" if is_online else "Offline"
    st.title(f"🌍 Hey {user_name}!")
    st.caption(f"Your AI companion — {status_emoji} {status_text} | Upload PDFs, images, or text files and ask anything.")

    if st.session_state.memory_manager is None:
        st.session_state.memory_manager = get_memory_manager()

    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader(f"👤 {user_name}")

        with st.expander("🔐 Account"):
            if st.button("💾 Save Session", use_container_width=True):
                if username:
                    _save_user_session(username)
                    st.success("Session saved!")

            if st.button("📤 Export My Data", use_container_width=True):
                if username:
                    exported = export_user_data(username)
                    if exported:
                        st.download_button(
                            label="Download JSON",
                            data=exported,
                            file_name=f"{username}_export.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    else:
                        st.error("Could not export data.")

            st.divider()
            if st.button("🚪 Log Out", use_container_width=True):
                if username:
                    _save_user_session(username)
                st.session_state.logged_in = False
                st.session_state.user_name = None
                st.session_state.user_info = None
                st.session_state.messages = []
                st.session_state.uploaded_files = []
                st.rerun()

        st.divider()

        if is_online:
            st.success("🌐 Online — Web search enabled")
        else:
            st.warning("📴 Offline — Using local knowledge only")

        mm = st.session_state.model_manager
        status = "Unknown"; mtype = "Unknown"
        if mm.model is not None:
            status = "Active"; mtype = mm.model_path
        st.info(f"Model: {status}\n{mtype[:40] if mtype else ''}")

        wm = st.session_state.world_model
        training_count = len(wm.training_data)
        if training_count > 0:
            st.divider()
            st.subheader("🧠 Self-Learning")
            st.write(f"Training data collected: {training_count} points")
            if st.button("💾 Export Training Data", use_container_width=True):
                count = wm.export_training_dataset("./data/self_training.json")
                st.success(f"Exported {count} training examples!")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            if st.session_state.memory_manager:
                try:
                    st.session_state.memory_manager.document_store = None
                    st.session_state.memory_manager._document_metadata = []
                except Exception:
                    pass
            if username:
                _save_user_session(username)
            st.rerun()

        if st.session_state.uploaded_files:
            st.divider()
            st.subheader("📄 Your Documents")
            for f in st.session_state.uploaded_files:
                st.write(f"- {f['name']} ({f.get('chunks', '?')} chunks)")

    if not st.session_state.initialized:
        st.info("Initializing model... (first time may download ~270MB)")
        try:
            mm = st.session_state.model_manager
            mm.load_model_sync()
            st.session_state.initialized = True
            st.success(f"Loaded: {mm.model_path}")
        except Exception as e:
            logger.error(f"Init failed: {e}")
            st.error(f"Model load failed: {e}")
            st.session_state.initialized = True

    if not st.session_state.messages:
        features = [
            "📄 Upload PDFs, images, or text files — I'll read and answer questions about them",
            "🔢 Solve math problems step-by-step",
            "🌐 Auto web search when online (no setup needed)",
            "🧠 Self-evaluating and self-improving responses",
            "🤖 All agents working together (Generator, Critic, Optimizer, Planner)",
            "⚡ Dynamic complexity adjustment — no manual settings needed",
        ]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; color: white;">
            <h3>✨ Welcome, {user_name}!</h3>
            <p>I'm your AI companion. Here's what I can do:</p>
            <ul>
                {''.join(f'<li>{f}</li>' for f in features)}
            </ul>
            <p style="margin-bottom: 0;"><strong>Just type below or upload a file to get started!</strong></p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("files"):
                with st.expander("📎 Attached Files"):
                    for f in msg["files"]:
                        st.write(f"- {f}")
            if msg.get("metadata"):
                with st.expander("🧠 Thinking Process"):
                    meta = msg["metadata"]
                    if meta.get("complexity"):
                        st.markdown(f"**Complexity:** `{meta['complexity']}`")
                    if meta.get("auto_config"):
                        st.caption(f"Auto-configured: Depth={meta.get('depth', 'auto')}, Steps={meta.get('steps', 'auto')}")
                    st.json({k: v for k, v in meta.items() if k not in ["thought_process", "tool_results"]})
                    if meta.get("self_eval"):
                        score = float(meta["self_eval"])
                        color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
                        st.markdown(f"**Self-Evaluation Score:** :{color}[{score:.2f}]")
                    if meta.get("tool_results"):
                        with st.expander("🔧 Tool Results"):
                            for tr in meta["tool_results"]:
                                st.code(tr)

    st.divider()
    uploaded_files = st.file_uploader(
        "📎 Upload documents (PDF, TXT, MD, PNG, JPG)",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload files to ground my answers in your documents.",
    )

    if uploaded_files:
        mem_mgr = st.session_state.memory_manager
        newly_processed = []
        for uf in uploaded_files:
            if any(p["name"] == uf.name for p in st.session_state.uploaded_files):
                continue
            with st.spinner(f"📖 Reading {uf.name}..."):
                result = process_uploaded_file(uf, mem_mgr)
                if result.get("success"):
                    st.session_state.uploaded_files.append({
                        "name": uf.name,
                        "chunks": result.get("chunks_ingested", 0),
                        "text_length": result.get("text_length", 0),
                    })
                    newly_processed.append(uf.name)
                else:
                    st.error(f"Failed to process {uf.name}: {result.get('error', 'Unknown error')}")
        if newly_processed:
            st.success(f"✅ Processed: {', '.join(newly_processed)}")
            st.rerun()

    prompt = st.chat_input(f"What would you like to talk about, {user_name}?")
    if prompt:
        document_context = ""
        if st.session_state.memory_manager and st.session_state.uploaded_files:
            with st.spinner("🔍 Searching your documents..."):
                doc_ctx = st.session_state.memory_manager.get_document_context(prompt, top_k=5)
                if doc_ctx:
                    document_context = doc_ctx

        with st.chat_message("user"):
            st.markdown(prompt)
            if st.session_state.uploaded_files:
                st.caption(f"📎 Grounded in {len(st.session_state.uploaded_files)} document(s)")

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "files": [f["name"] for f in st.session_state.uploaded_files],
        })

        with st.chat_message("assistant"):
            with st.spinner("💭 Thinking... (all agents collaborating)"):
                try:
                    wm = st.session_state.world_model
                    context = {"user_name": user_name}
                    if document_context:
                        context["document_context"] = document_context

                    if st.session_state.memory_manager:
                        mem_ctx = st.session_state.memory_manager.get_relevant_context(prompt, top_k=2)
                        if mem_ctx:
                            context["retrieved_knowledge"] = mem_ctx

                    result = _run_async(_fast_direct_generate(prompt, wm, st.session_state.model_manager, context))

                    content = result.get("content", "No response generated.")
                    st.markdown(content)

                    if result.get("self_eval"):
                        se = result["self_eval"]
                        color = "green" if se > 0.7 else "orange" if se > 0.5 else "red"
                        st.caption(f"🧠 Self-Evaluation: :{color}[{se:.2f}/1.0] | Complexity: {result.get('complexity', 'unknown')}")

                    if result.get("tool_results"):
                        with st.expander("🔧 Tool Results"):
                            for tr in result["tool_results"]:
                                st.code(tr)

                    metadata = {
                        "thinking_steps": result.get("thinking_steps", 0),
                        "scenarios": result.get("scenarios", 0),
                        "strategy": result.get("strategy", "N/A"),
                        "self_eval": f"{result.get('self_evaluation_score', 0):.2f}",
                        "causal": f"{result.get('causal_score', 0):.2f}",
                        "physics": f"{result.get('physics_score', 0):.2f}",
                        "complexity": result.get("complexity", "unknown"),
                        "depth": result.get("depth", wm.imagination_depth),
                        "steps": result.get("steps", wm.thinking_steps),
                        "auto_config": True,
                        "tool_results": result.get("tool_results", []),
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content,
                        "metadata": metadata,
                    })

                    if st.session_state.memory_manager:
                        st.session_state.memory_manager.add_entry(
                            user_query=prompt, output=content, critique="",
                            scores={"overall": result.get("self_evaluation_score", 0.5)},
                            system_prompt="", iteration=1,
                        )

                    # Auto-save after each message exchange
                    if username:
                        _save_user_session(username)

                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    st.error(f"Error: {e}")
                    fallback = "I'm sorry, something went wrong. Can you try rephrasing that?"
                    st.markdown(fallback)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback,
                        "metadata": {"error": str(e)},
                    })


async def _simple_generate(prompt, world_model, model_manager, context):
    thoughts = await world_model.think(prompt, context)
    scenarios = await world_model.imagine_scenarios(prompt, context)
    return await world_model.generate_response(
        prompt, scenarios, thoughts, model_manager,
        document_context=context.get("document_context"),
        context=context
    )


async def _fast_direct_generate(prompt, world_model, model_manager, context):
    """Full multi-agent pipeline: Generator -> Critic -> Optimizer -> Planner -> Self-Eval."""
    gen = GeneratorAgent(model_path=model_manager.model_path)
    gen_response = await gen.execute(prompt, context=context)
    
    if not gen_response.success:
        return {"content": f"Generation failed: {gen_response.error}", "self_evaluation_score": 0.0}
    
    initial_output = gen_response.content
    
    crit = CriticAgent(model_path=model_manager.model_path)
    crit_response = await crit.execute(initial_output, context={"user_query": prompt})
    
    eval_data = await world_model.self_evaluate_and_improve(prompt, initial_output, model_manager)
    
    if eval_data.get("needs_improvement", False):
        opt = PromptOptimizerAgent(model_path=model_manager.model_path)
        opt_response = await opt.execute("", context={
            "current_prompt": "",
            "critique": crit_response.content if crit_response.success else "",
            "scores": crit_response.metadata.get("scores", {}) if crit_response.success else {},
        })
        
        planner = PlannerAgent(model_path=model_manager.model_path)
        plan_response = await planner.execute(f"Improve response to: {prompt}")
        
        thoughts = await world_model.think(prompt, context)
        scenarios = await world_model.imagine_scenarios(prompt, context)
        
        improved_context = context.copy()
        improved_context["critique"] = crit_response.content if crit_response.success else ""
        improved_context["improvement_plan"] = plan_response.content if plan_response.success else ""
        
        result = await world_model.generate_response(
            prompt, scenarios, thoughts, model_manager,
            document_context=context.get("document_context"),
            context=improved_context
        )
        
        re_eval = await world_model.self_evaluate_and_improve(prompt, result["content"], model_manager)
        result["self_evaluation_score"] = re_eval["overall"]
        result["causal_score"] = re_eval["causal"]
        result["physics_score"] = re_eval["physics"]
        result["improved"] = True
        result["original_score"] = eval_data["overall"]
    else:
        thoughts = await world_model.think(prompt, context)
        scenarios = await world_model.imagine_scenarios(prompt, context)
        result = await world_model.generate_response(
            prompt, scenarios, thoughts, model_manager,
            document_context=context.get("document_context"),
            context=context
        )
        result["self_evaluation_score"] = eval_data["overall"]
        result["causal_score"] = eval_data["causal"]
        result["physics_score"] = eval_data["physics"]
        result["improved"] = False
    
    complexity = world_model._compute_complexity(prompt)
    result["complexity"] = complexity["complexity"]
    result["depth"] = world_model.imagination_depth
    result["steps"] = world_model.thinking_steps
    
    await gen.close()
    await crit.close()
    
    return result


if __name__ == "__main__":
    run_app()

