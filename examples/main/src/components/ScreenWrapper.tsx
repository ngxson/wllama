export default function ScreenWrapper({
  children,
  fitScreen,
}: {
  children: any;
  fitScreen?: boolean;
}) {
  return (
    <div className="overflow-y-auto h-full w-full">
      <div
        className={`w-[40rem] max-w-full ${fitScreen ? 'h-full' : 'h-auto overflow-hidden'} px-4 flex flex-col mx-auto`}
      >
        {children}
      </div>
    </div>
  );
}
