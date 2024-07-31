function App() {
  return (
    <div className="flex flex-col drawer h-screen w-screen overflow-hidden">

      <div className="navbar bg-base-100">
        <div className="flex-none">
          <label htmlFor="my-drawer-2" className="btn btn-square btn-ghost flex lg:hidden">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              className="inline-block h-5 w-5 stroke-current">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
          </label>
        </div>
        <a className="btn btn-ghost text-xl">wllama</a>
      </div>

      <div className="grow flex flex-row lg:drawer-open h-[calc(100vh-4rem)]">
        <input id="my-drawer-2" type="checkbox" className="drawer-toggle" />

        <div className="drawer-side h-screen lg:h-[calc(100vh-4rem)] z-50">
          <label htmlFor="my-drawer-2" aria-label="close sidebar" className="drawer-overlay block lg:hidden"></label>

          <div className="h-screen lg:max-h-[calc(100vh-4rem)] flex flex-col text-base-content bg-base-200">
            <div className="grow w-80 overflow-auto p-4">
              <ul className="grow menu">
                {/* Sidebar content here */}
                <li><a>Sidebar Item 1</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
              </ul>
            </div>

            <div className="w-80 p-4 pt-0">
              <div className="divider"></div>

              <ul className="menu">
                {/* Sidebar content here */}
                <li><a>Sidebar Item 1</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
                <li><a>Sidebar Item 2</a></li>
              </ul>
            </div>
          </div>
        </div>

        <div className="drawer-content mx-auto">
          <div className="w-[40rem] max-w-full h-full px-4 flex flex-col">
            <div className="chat-messages grow overflow-auto">

              <div className="h-10" />


              <div className="chat chat-start">
                <div className="chat-image avatar">
                  <div className="w-10 rounded-full">
                    <img
                      alt="Tailwind CSS chat bubble component"
                      src="https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp" />
                  </div>
                </div>
                <div className="chat-bubble">It was said that you would, destroy the Sith, not join them.</div>
              </div>
              <div className="chat chat-start">
                <div className="chat-image avatar">
                  <div className="w-10 rounded-full">
                    <img
                      alt="Tailwind CSS chat bubble component"
                      src="https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp" />
                  </div>
                </div>
                <div className="chat-bubble">It was you who would bring balance to the Force</div>
              </div>
              <div className="chat chat-start">
                <div className="chat-image avatar">
                  <div className="w-10 rounded-full">
                    <img
                      alt="Tailwind CSS chat bubble component"
                      src="https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp" />
                  </div>
                </div>
                <div className="chat-bubble">Not leave it in Darkness</div>
              </div>


            </div>
            <div className="input-message py-4">
              <textarea className="textarea textarea-bordered w-full" placeholder="Your message"></textarea>
            </div>
          </div>
        </div>
      </div>


    </div>
  );
}

export default App;
