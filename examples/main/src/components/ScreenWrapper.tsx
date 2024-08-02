export default function ScreenWrapper({ children }: { children: any }) {
  return (
    <div className="overflow-y-auto h-full w-full">
      <div className="w-[40rem] max-w-full h-auto overflow-hidden px-4 flex flex-col mx-auto">
        {children}
      </div>
    </div>
  );
}